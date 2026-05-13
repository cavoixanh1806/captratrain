"""
train.py
========
Module huấn luyện (fine-tune) mô hình TrOCR trên dataset CAPTCHA.

Sử dụng Seq2SeqTrainer của Hugging Face để fine-tune
VisionEncoderDecoderModel từ checkpoint microsoft/trocr-small-printed.

Metric đánh giá: CER (Character Error Rate) — tỷ lệ lỗi ký tự.
CER = (S + D + I) / N
  S = số ký tự bị thay thế sai
  D = số ký tự bị xóa
  I = số ký tự bị chèn thêm
  N = tổng số ký tự trong ground truth
CER càng thấp → model càng chính xác (CER = 0 là hoàn hảo).

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
import evaluate

from dataset import create_datasets

# ─── Cấu hình logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hằng số cấu hình ────────────────────────────────────────────────────────
MODEL_CHECKPOINT: str = "microsoft/trocr-base-printed"  # Base chính xác hơn small
OUTPUT_DIR: str = "./captcha_trocr_model"
REAL_DATA_DIR: str = "data"

# Hyperparameters — Tối ưu cho RTX 3060 12GB VRAM, giới hạn 10GB
# i5-12400F (6c/12t) + 16GB RAM (dùng tối đa 10GB)
BATCH_SIZE: int = 32         # RTX 3060 xử lý batch lớn hiệu quả
LEARNING_RATE: float = 2e-5  # Giảm xuống để học ổn định hơn, giảm overfit
NUM_EPOCHS: int = 30         # GPU nhanh hơn, có thể train nhiều epochs
SAVE_STEPS: int = 50
EVAL_STEPS: int = 50
MAX_TARGET_LENGTH: int = 8   # CAPTCHA cố định 5 ký tự + special tokens


def compute_cer_metric(
    processor: TrOCRProcessor,
    cer_metric: Any,
) -> Any:
    """Tạo hàm compute_metrics để tính CER trong quá trình evaluation.

    Hàm này được truyền vào Seq2SeqTrainer. Sau mỗi epoch validation,
    Trainer sẽ gọi hàm này với EvalPrediction chứa predictions và labels.

    Args:
        processor: TrOCRProcessor để decode token IDs thành text.
        cer_metric: Metric CER từ thư viện `evaluate`.

    Returns:
        Hàm compute_metrics(eval_pred) → dict{"cer": float}.
    """

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Tính CER từ predictions và labels.

        Args:
            eval_pred: EvalPrediction object với:
                - predictions: numpy array token IDs dự đoán, shape (N, seq_len).
                - label_ids: numpy array token IDs ground truth, shape (N, seq_len).

        Returns:
            Dict {"cer": float} — giá trị CER trung bình trên tập val.
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

        # Tính CER: so sánh từng cặp (prediction, ground_truth)
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer_score}

    return compute_metrics


def setup_model(processor: TrOCRProcessor) -> VisionEncoderDecoderModel:
    """Load và cấu hình VisionEncoderDecoderModel từ checkpoint TrOCR.

    Các tham số decoder cần được cấu hình thủ công để model hoạt động đúng:
      - decoder_start_token_id: Token bắt đầu sinh chuỗi (BOS token).
      - eos_token_id: Token kết thúc chuỗi (EOS token).
      - pad_token_id: Token padding (dùng để fill chuỗi ngắn hơn max_length).
      - vocab_size: Kích thước từ điển của decoder.

    Args:
        processor: TrOCRProcessor đã được load.

    Returns:
        VisionEncoderDecoderModel đã được cấu hình.
    """
    logger.info(f"Đang load model từ checkpoint: {MODEL_CHECKPOINT}")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

    # ── Cấu hình các token đặc biệt cho decoder ──────────────────────────────
    # Đây là bước BẮT BUỘC khi fine-tune VisionEncoderDecoderModel.
    # Nếu bỏ qua, model sẽ không biết khi nào bắt đầu/kết thúc sinh chuỗi.
    tokenizer = processor.tokenizer

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # vocab_size phải khớp với tokenizer để embedding layer hoạt động đúng
    model.config.vocab_size = model.config.decoder.vocab_size

    # ── Cấu hình generation thông qua GenerationConfig (cách đúng ở TF 5.x) ──
    # Trước đây đặt trực tiếp vào model.config đã bị deprecated.
    model.generation_config = GenerationConfig(
        max_length=MAX_TARGET_LENGTH,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
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
        warmup_steps=200,           # Warm up 200 steps đầu để ổn định training
        weight_decay=0.01,          # L2 regularization

        # ── Epochs và evaluation ──────────────────────────────────────────────
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",      # Evaluate sau mỗi eval_steps (TF 5.x)
        eval_steps=EVAL_STEPS,

        # ── Lưu checkpoint ────────────────────────────────────────────────────
        save_steps=SAVE_STEPS,
        save_total_limit=3,         # Chỉ giữ 3 checkpoint gần nhất
        load_best_model_at_end=True,  # Load model tốt nhất khi kết thúc
        metric_for_best_model="cer",  # Chọn model dựa trên CER thấp nhất
        greater_is_better=False,    # CER thấp hơn = tốt hơn

        # ── Logging ───────────────────────────────────────────────────────────
        logging_steps=100,
        report_to="none",           # Tắt wandb/tensorboard

        # ── Seq2Seq specific ──────────────────────────────────────────────────
        predict_with_generate=True,  # Dùng generate() khi evaluate (không dùng logits)
        generation_max_length=MAX_TARGET_LENGTH,

        # ── Tối ưu bộ nhớ ─────────────────────────────────────────────────────
        fp16=torch.cuda.is_available(),  # Mixed precision FP16 trên RTX 3060
        dataloader_num_workers=4,   # i5-12400F có 6c/12t, dùng 4 workers
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory khi có GPU
        label_names=["labels"],     # Khai báo tên labels cho Trainer TF 5.x
    )


def main(use_real_data: bool = False, combine: bool = False) -> None:
    """Hàm chính: load data, setup model và bắt đầu fine-tuning.

    Args:
        use_real_data: True để dùng Real Data từ thư mục data/.
        combine: True để kết hợp cả Synthetic và Real Data.
    """
    # ── Kiểm tra GPU ──────────────────────────────────────────────────────────
    # Tối ưu CPU: dùng hết threads của i5-12400F (6c/12t)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(6)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        # Giới hạn VRAM tối đa 10GB để chừa RAM cho các tác vụ khác
        total_vram = torch.cuda.get_device_properties(0).total_memory
        max_vram = int(10 * 1024 ** 3)  # 10GB
        if total_vram > max_vram:
            torch.cuda.set_per_process_memory_fraction(max_vram / total_vram)
            logger.info(f"🚀 Sử dụng GPU: {device_name} (giới hạn 10GB VRAM)")
        else:
            logger.info(f"🚀 Sử dụng GPU: {device_name}")
    else:
        logger.warning(
            "⚠️  Không tìm thấy GPU (CUDA). Sẽ chạy trên CPU — rất chậm! "
            "Khuyến nghị dùng máy có GPU để huấn luyện."
        )

    # ── Load Processor ────────────────────────────────────────────────────────
    logger.info(f"Đang load TrOCRProcessor từ: {MODEL_CHECKPOINT}")
    processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)

    # ── Load Dataset ──────────────────────────────────────────────────────────
    if combine:
        # Kết hợp Synthetic + Real Data
        from torch.utils.data import ConcatDataset

        logger.info("Chế độ: Kết hợp Synthetic + Real Data")
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
        )
        train_dataset = ConcatDataset([syn_train, real_train])
        val_dataset = ConcatDataset([syn_val, real_val])

    else:
        mode = "Real Data" if use_real_data else "Synthetic Data"
        logger.info(f"Chế độ: {mode}")
        train_dataset, val_dataset = create_datasets(
            processor,
            use_real_data=use_real_data,
            real_data_dir=REAL_DATA_DIR,
            max_target_length=MAX_TARGET_LENGTH,
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples:   {len(val_dataset)}")

    # ── Setup Model ───────────────────────────────────────────────────────────
    model = setup_model(processor)

    # ── Load CER Metric ───────────────────────────────────────────────────────
    # `evaluate` là thư viện của Hugging Face để tính các metric chuẩn
    logger.info("Đang load metric CER...")
    cer_metric = evaluate.load("cer")
    compute_metrics = compute_cer_metric(processor, cer_metric)

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = get_training_args(OUTPUT_DIR)

    # ── Khởi tạo Seq2SeqTrainer ───────────────────────────────────────────────
    # default_data_collator: gộp các sample thành batch tensor
    # Với Seq2Seq, nó tự động xử lý padding cho labels
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,  # Thay thế cho tokenizer (deprecated) trong TF 5.x
        # EarlyStopping: dừng sớm nếu CER không cải thiện trong 3 lần eval liên tiếp
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Bắt đầu Fine-tuning ───────────────────────────────────────────────────
    logger.info("🏋️  Bắt đầu fine-tuning...")
    trainer.train()

    # ── Lưu Model và Processor ────────────────────────────────────────────────
    logger.info(f"💾 Lưu model tốt nhất vào: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    logger.info("🎉 Huấn luyện hoàn tất!")
    logger.info(f"   Model đã lưu tại: {OUTPUT_DIR}")
    logger.info(f"   Chạy inference: python inference.py <đường_dẫn_ảnh>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR trên dataset CAPTCHA"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Dùng Real Data từ thư mục data/ (cần có metadata.csv)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Kết hợp cả Synthetic Data và Real Data",
    )
    args = parser.parse_args()

    main(use_real_data=args.use_real_data, combine=args.combine)
