"""
train.py
========
Module huấn luyện (fine-tune) mô hình TrOCR trên dataset CAPTCHA.

Sử dụng Seq2SeqTrainer của Hugging Face để fine-tune
VisionEncoderDecoderModel từ checkpoint microsoft/trocr-base-printed.

Metric đánh giá:
  - CER (Character Error Rate): tỷ lệ lỗi ký tự — càng thấp càng tốt.
  - Exact Match Accuracy: % ảnh đoán đúng hoàn toàn — metric thực tế nhất
    vì CAPTCHA chỉ cần 1 ký tự sai là fail.

FIX:
  - Bỏ no_repeat_ngram_size=3 (cấm ký tự lặp như "AA123" → sai kết quả).
  - Giảm length_penalty từ 2.0 xuống 1.0 (CAPTCHA ngắn cố định 5 ký tự).
  - Tăng EarlyStopping patience từ 3 lên 8 (tránh underfitting với 500 ảnh).
  - Thêm Exact Match Accuracy vào compute_metrics.

Cách chạy:
    # Dùng Synthetic Data:
    python train.py

    # Dùng Real Data (sau khi đã dán nhãn xong):
    python train.py --use-real-data

    # Kết hợp cả hai:
    python train.py --use-real-data --combine
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Tắt HTTP request logs của huggingface_hub (quá nhiều noise)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    GenerationConfig,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

from dataset import create_datasets

# ─── Cấu hình logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hằng số cấu hình ────────────────────────────────────────────────────────
MODEL_CHECKPOINT: str = "microsoft/trocr-base-printed"
OUTPUT_DIR: str = "./captcha_trocr_model"
REAL_DATA_DIR: str = "data"

# Hyperparameters — Tối ưu cho 500 ảnh real data
# RTX 3060 8GB VRAM, i5-12400F (6c/12t) + 16GB RAM
# 500 * 0.8 = 400 train, batch=16 → 25 steps/epoch, 50 epochs → 1250 total steps
BATCH_SIZE: int = 16
LEARNING_RATE: float = 5e-5
NUM_EPOCHS: int = 50
SAVE_STEPS: int = 25         # Save mỗi ~1 epoch
EVAL_STEPS: int = 25         # Eval mỗi ~1 epoch
MAX_TARGET_LENGTH: int = 8   # [BOS] + 5 ASCII chars + [EOS] = 7 tokens, 8 dư sức

# FIX: Tăng patience từ 3 lên 8 — tránh dừng quá sớm với 500 ảnh
# patience=3 → dừng sau 75 steps (~3 epochs) nếu CER không cải thiện
# patience=8 → dừng sau 200 steps (~8 epochs) — hợp lý hơn
EARLY_STOPPING_PATIENCE: int = 8


def compute_cer_metric(
    processor: TrOCRProcessor,
    cer_metric: Any,
) -> Any:
    """Tạo hàm compute_metrics để tính CER và Exact Match Accuracy.

    FIX: Thêm exact_match — metric thực tế nhất cho CAPTCHA.
    CAPTCHA chỉ cần 1 ký tự sai là fail hoàn toàn, nên CER một mình
    không phản ánh đúng hiệu quả thực tế.

    Args:
        processor: TrOCRProcessor để decode token IDs thành text.
        cer_metric: Metric CER từ thư viện `evaluate`.

    Returns:
        Hàm compute_metrics(eval_pred) → dict{"cer": float, "exact_match": float}.
    """

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Tính CER và Exact Match từ predictions và labels.

        Args:
            eval_pred: EvalPrediction object với:
                - predictions: numpy array token IDs dự đoán.
                - label_ids: numpy array token IDs ground truth.

        Returns:
            Dict {"cer": float, "exact_match": float}.
        """
        pred_ids, label_ids = eval_pred

        # Xử lý trường hợp predictions là tuple (logits, ...)
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Lấy token ID có xác suất cao nhất nếu là logits (3D array)
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        # Thay -100 (padding đã mask) bằng pad_token_id để decode được
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode token IDs → chuỗi text, bỏ qua special tokens
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize: strip + uppercase (CAPTCHA luôn uppercase)
        pred_str = [p.strip().upper() for p in pred_str]
        label_str = [l.strip().upper() for l in label_str]

        # CER: tỷ lệ lỗi ký tự
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)

        # Exact Match: % ảnh đoán đúng hoàn toàn
        # Đây là metric quan trọng nhất cho CAPTCHA — 1 ký tự sai = fail
        exact_match = sum(
            p == r for p, r in zip(pred_str, label_str)
        ) / max(len(pred_str), 1)

        return {
            "cer": cer_score,
            "exact_match": exact_match,
        }

    return compute_metrics


def setup_model(processor: TrOCRProcessor) -> VisionEncoderDecoderModel:
    """Load và cấu hình VisionEncoderDecoderModel từ checkpoint TrOCR.

    FIX:
      - Bỏ no_repeat_ngram_size: CAPTCHA có thể có ký tự lặp ("AA123", "11KKK")
        → no_repeat_ngram_size=3 sẽ cấm model sinh 3 token giống nhau liên tiếp
        → kết quả sai với CAPTCHA có ký tự lặp.
      - Giảm length_penalty từ 2.0 xuống 1.0: CAPTCHA luôn đúng 5 ký tự,
        không cần penalty để ưu tiên chuỗi dài hơn.

    Args:
        processor: TrOCRProcessor đã được load.

    Returns:
        VisionEncoderDecoderModel đã được cấu hình.
    """
    logger.info(f"Đang load model từ checkpoint: {MODEL_CHECKPOINT}")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

    # ── Cấu hình các token đặc biệt cho decoder ──────────────────────────────
    tokenizer = processor.tokenizer

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # ── Cấu hình generation ───────────────────────────────────────────────────
    # FIX: Bỏ no_repeat_ngram_size — CAPTCHA có thể có ký tự lặp
    # FIX: length_penalty=1.0 thay vì 2.0 — CAPTCHA ngắn cố định 5 ký tự
    model.generation_config = GenerationConfig(
        max_length=MAX_TARGET_LENGTH,
        early_stopping=True,
        length_penalty=1.0,          # FIX: 1.0 thay vì 2.0
        num_beams=4,
        # no_repeat_ngram_size bị bỏ — CAPTCHA có thể có ký tự lặp
        decoder_start_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info("✅ Model đã được cấu hình thành công.")
    return model


def get_training_args(output_dir: str) -> Seq2SeqTrainingArguments:
    """Tạo Seq2SeqTrainingArguments với các hyperparameter đã cấu hình.

    Args:
        output_dir: Thư mục lưu checkpoints và model cuối cùng.

    Returns:
        Seq2SeqTrainingArguments đã được cấu hình.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,

        # ── Batch size ────────────────────────────────────────────────────────
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        # ── Learning rate và schedule ─────────────────────────────────────────
        learning_rate=LEARNING_RATE,
        warmup_steps=100,           # ~4 epochs = ~8% total steps (1250)
        weight_decay=0.01,          # L2 regularization

        # ── Epochs và evaluation ──────────────────────────────────────────────
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # ── Lưu checkpoint ────────────────────────────────────────────────────
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",  # FIX: dùng exact_match thay vì cer
        greater_is_better=True,               # FIX: exact_match cao hơn = tốt hơn

        # ── Logging ───────────────────────────────────────────────────────────
        logging_steps=25,
        report_to="none",

        # ── Seq2Seq specific ──────────────────────────────────────────────────
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # ── Tối ưu bộ nhớ ─────────────────────────────────────────────────────
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,   # Windows không pickle được nested fn với workers>0
        dataloader_pin_memory=torch.cuda.is_available(),
        label_names=["labels"],
    )


def main(
    use_real_data: bool = False,
    combine: bool = False,
    preprocess_method: str | None = None,
    augment: bool = False,
) -> None:
    """Hàm chính: load data, setup model và bắt đầu fine-tuning.

    Args:
        use_real_data: True để dùng Real Data từ thư mục data/.
        combine: True để kết hợp cả Synthetic và Real Data.
        preprocess_method: Phương pháp preprocessing.
        augment: True để bật data augmentation cho training.
    """
    # ── Kiểm tra GPU ──────────────────────────────────────────────────────────
    torch.set_num_threads(12)
    torch.set_num_interop_threads(6)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        max_vram = int(10 * 1024 ** 3)  # 10GB
        if total_vram > max_vram:
            torch.cuda.set_per_process_memory_fraction(max_vram / total_vram)
            logger.info(f"🚀 Sử dụng GPU: {device_name} (giới hạn 10GB VRAM)")
        else:
            logger.info(f"🚀 Sử dụng GPU: {device_name}")
    else:
        logger.warning(
            "⚠️  Không tìm thấy GPU (CUDA). Sẽ chạy trên CPU — rất chậm!"
        )

    # ── Load Processor ────────────────────────────────────────────────────────
    logger.info(f"Đang load TrOCRProcessor từ: {MODEL_CHECKPOINT}")
    processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)

    # ── Load Dataset ──────────────────────────────────────────────────────────
    if combine:
        from torch.utils.data import ConcatDataset

        logger.info("Mode: Synthetic + Real Data")
        syn_train, syn_val = create_datasets(
            processor,
            use_real_data=False,
            max_target_length=MAX_TARGET_LENGTH,
        )
        real_train, real_val = create_datasets(
            processor,
            use_real_data=True,
            real_data_dir=REAL_DATA_DIR,
            max_target_length=MAX_TARGET_LENGTH,
            preprocess_method=preprocess_method,
            augment=augment,
        )
        train_dataset = ConcatDataset([syn_train, real_train])
        val_dataset = ConcatDataset([syn_val, real_val])

    else:
        mode = "Real Data" if use_real_data else "Synthetic Data"
        if preprocess_method:
            mode += f" + Preprocessing({preprocess_method})"
        if augment:
            mode += " + Augmentation"
        logger.info(f"Mode: {mode}")
        train_dataset, val_dataset = create_datasets(
            processor,
            use_real_data=use_real_data,
            real_data_dir=REAL_DATA_DIR,
            max_target_length=MAX_TARGET_LENGTH,
            preprocess_method=preprocess_method,
            augment=augment,
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples:   {len(val_dataset)}")

    # ── Setup Model ───────────────────────────────────────────────────────────
    model = setup_model(processor)

    # ── Load CER Metric ───────────────────────────────────────────────────────
    logger.info("Đang load metric CER...")
    cer_metric = hf_evaluate.load("cer")
    compute_metrics = compute_cer_metric(processor, cer_metric)

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = get_training_args(OUTPUT_DIR)

    # ── Khởi tạo Seq2SeqTrainer ───────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        # FIX: Tăng patience từ 3 lên 8 — tránh dừng quá sớm với 500 ảnh
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )],
    )

    # ── Bắt đầu Fine-tuning ───────────────────────────────────────────────────
    logger.info("🏋️  Bắt đầu fine-tuning...")
    trainer.train()

    # ── Lưu Model và Processor ────────────────────────────────────────────────
    logger.info(f"[SAVE] Best model saved to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    logger.info("[DONE] Training complete!")
    logger.info(f"   Model saved at: {OUTPUT_DIR}")
    logger.info(f"   Run inference: python inference.py <image_path>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on CAPTCHA dataset"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use Real Data from data/ directory (requires metadata.csv)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine Synthetic + Real Data",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (rotation, brightness, blur, elastic)",
    )
    args = parser.parse_args()

    main(
        use_real_data=args.use_real_data,
        combine=args.combine,
        preprocess_method="unet",
        augment=args.augment,
    )
