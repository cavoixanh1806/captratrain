"""
self_train.py
==============
Self-training (pseudo-labeling) round 2 — theo research doc Phase 4.

Mục đích:
    Sau khi train round 1 đạt ~80-85%, dùng model predict TẤT CẢ ảnh real
    với confidence. Chỉ giữ những predictions có:
      1. Confidence cao (≥ threshold, default 0.95)
      2. Khớp với label gốc trong metadata.csv (verify ground truth)

Những samples này → "high quality" set → fine-tune thêm vài epochs với:
    - LR thấp hơn (1e-4)
    - Weight cao hơn cho real (vì đây là "verified" data)
    - Vẫn giữ synthetic làm regularizer

Theo research doc, self-training có thể boost +5-10% accuracy.

Usage:
    # Sau khi đã train round 1:
    python self_train.py
    python self_train.py --confidence 0.92 --epochs 15
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from crnn_model import (
    CRNN,
    CTC_BLANK_INDEX,
    NUM_CLASSES,
    decode_greedy,
    load_crnn,
    save_crnn,
    export_onnx,
)
from dataset_crnn import CRNNCaptchaDataset, collate_fn, create_crnn_datasets
from inference_crnn import CRNNCaptchaSolver
from train_crnn import (
    CHECKPOINT_PATH,
    LAST_CHECKPOINT_PATH,
    ONNX_PATH,
    build_warmup_cosine_scheduler,
    compute_metrics,
    train_one_epoch,
    validate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE: float = 0.95    # Chỉ giữ predictions ≥ 95% confidence
DEFAULT_EPOCHS: int = 15            # Round 2 ngắn hơn round 1
DEFAULT_LR: float = 1e-4            # 10× thấp hơn round 1 (fine-tune)
DEFAULT_BATCH_SIZE: int = 64
WARMUP_STEPS: int = 50              # Warmup ngắn vì đã có pretrained weights
HIGH_CONF_OUTPUT_CSV: str = "data/_high_confidence.csv"


def select_high_confidence_samples(
    solver: CRNNCaptchaSolver,
    metadata_path: Path,
    image_dir: Path,
    confidence_threshold: float,
    batch_size: int = 64,
) -> tuple[Path, dict]:
    """Predict tất cả real samples, lọc theo confidence + ground-truth match.

    Args:
        solver: CRNNCaptchaSolver đã load round 1 model.
        metadata_path: data/metadata.csv với label gốc.
        image_dir: thư mục data/.
        confidence_threshold: ngưỡng confidence (0-1).
        batch_size: batch size cho inference.

    Returns:
        Tuple (path đến filtered metadata CSV, stats dict).
    """
    df = pd.read_csv(metadata_path, dtype=str).dropna()
    df["text"] = df["text"].str.strip().str.upper()
    df = df[df["text"].str.len() == 5].reset_index(drop=True)

    logger.info(f"Predicting on {len(df)} real samples...")

    image_paths = [str(image_dir / fn) for fn in df["filename"]]
    labels = df["text"].tolist()

    t = time.time()
    results = solver.solve_batch_with_confidence(image_paths, batch_size=batch_size)
    elapsed = time.time() - t
    logger.info(f"Inference done in {elapsed:.1f}s")

    # Lọc theo 2 điều kiện
    high_conf_rows = []
    correct_conf = []
    correct_low_conf = []
    wrong_conf = []
    for fn, label, (pred, conf) in zip(df["filename"], labels, results):
        is_correct = (pred == label)
        if conf >= confidence_threshold and is_correct:
            high_conf_rows.append({"filename": fn, "text": label})
            correct_conf.append(conf)
        elif is_correct:
            correct_low_conf.append(conf)
        else:
            wrong_conf.append(conf)

    stats = {
        "total": len(df),
        "high_confidence_correct": len(high_conf_rows),
        "low_confidence_correct": len(correct_low_conf),
        "wrong": len(wrong_conf),
        "selected_pct": len(high_conf_rows) / max(len(df), 1) * 100,
        "round1_exact_match": (len(correct_conf) + len(correct_low_conf)) / max(len(df), 1),
    }

    logger.info(f"Round 1 exact_match on real: {stats['round1_exact_match']*100:.2f}%")
    logger.info(
        f"High confidence ≥{confidence_threshold} AND correct: "
        f"{stats['high_confidence_correct']}/{stats['total']} "
        f"({stats['selected_pct']:.1f}%)"
    )
    logger.info(
        f"Correct but low confidence: {stats['low_confidence_correct']} "
        f"(model nói đúng nhưng không tự tin)"
    )
    logger.info(f"Wrong predictions: {stats['wrong']}")

    if high_conf_rows:
        avg_conf = sum(correct_conf) / len(correct_conf)
        logger.info(f"Avg confidence of selected: {avg_conf*100:.2f}%")

    # Save filtered metadata
    out_path = Path(HIGH_CONF_OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(high_conf_rows).to_csv(out_path, index=False)
    logger.info(f"Saved filtered metadata: {out_path}")

    return out_path, stats


def fine_tune_round2(
    model_path: str,
    high_conf_csv: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    image_dir: Path,
    use_synthetic: bool = True,
    synthetic_dir: str = "data/synthetic_crnn",
) -> None:
    """Fine-tune round 2 trên high-confidence samples.

    Strategy:
        - Train: high-conf real (verified) + synthetic (regularizer)
        - Val: KHÔNG dùng high-conf (vì đã thấy trong train), dùng val gốc
        - LR thấp (1e-4), epochs ít (15)
        - EMA + warmup ngắn
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device}")

    # ── Load round 1 model ────────────────────────────────────────────────────
    logger.info(f"Loading round 1 model: {model_path}")
    model = load_crnn(model_path, device=str(device))

    # ── Build datasets ────────────────────────────────────────────────────────
    # High-conf real: dùng làm train
    high_conf_ds = CRNNCaptchaDataset(
        image_dir=image_dir,
        metadata_path=high_conf_csv,
        augment=True,
    )
    logger.info(f"High-confidence real samples (train): {len(high_conf_ds)}")

    # Val từ split gốc (không bao gồm high-conf cho fair eval)
    # Dùng create_crnn_datasets với val_split=0.15 — val là 15% real ngẫu nhiên
    _, val_ds = create_crnn_datasets(
        real_data_dir=image_dir.parent if image_dir.name == "data" else "data",
        synthetic_dir=synthetic_dir,
        val_split=0.15,
        use_real=True,
        use_synthetic=False,  # Val chỉ real
        augment_train=False,
        seed=42,             # Same seed như round 1 → val giống nhau
    )
    logger.info(f"Val (real, original split): {len(val_ds)}")

    # Synthetic train (optional regularizer)
    train_parts: list = [high_conf_ds]
    if use_synthetic:
        syn_meta = Path(synthetic_dir) / "metadata.csv"
        if syn_meta.exists():
            syn_ds = CRNNCaptchaDataset(
                image_dir=synthetic_dir, metadata_path=syn_meta, augment=True,
            )
            # Subsample synthetic — không cần full 100K, chỉ làm regularizer
            n_syn = min(len(syn_ds), len(high_conf_ds) * 10)
            indices = torch.randperm(len(syn_ds))[:n_syn].tolist()
            syn_subset = Subset(syn_ds, indices)
            train_parts.append(syn_subset)
            logger.info(f"Synthetic regularizer: {n_syn} samples")
        else:
            logger.warning(f"No synthetic data at {syn_meta}, train trên real only")

    if len(train_parts) > 1:
        train_ds = torch.utils.data.ConcatDataset(train_parts)
    else:
        train_ds = train_parts[0]
    logger.info(f"Total train samples: {len(train_ds)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    criterion = nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps // 10),
        total_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed-precision (fp16) enabled")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        f"Round 2 config: epochs={epochs}, batch={batch_size}, lr={lr:.1e}, "
        f"warmup={min(WARMUP_STEPS, total_steps // 10)}, total_steps={total_steps}"
    )
    logger.info("-" * 90)

    # Initial val (round 1 baseline)
    val0 = validate(model, val_loader, criterion, device)
    logger.info(
        f"[BEFORE round 2] val: em={val0['val_exact_match']:.4f}, "
        f"cer={val0['val_cer']:.4f}"
    )

    best_val_em = val0["val_exact_match"]
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"R2 Epoch {epoch:2d}/{epochs} ({elapsed:.0f}s) lr={lr_now:.2e} | "
            f"train: loss={train_metrics['loss']:.4f} | "
            f"val: em={val_metrics['val_exact_match']:.4f} "
            f"cer={val_metrics['val_cer']:.4f}"
        )

        if val_metrics["val_exact_match"] > best_val_em:
            best_val_em = val_metrics["val_exact_match"]
            best_epoch = epoch
            save_crnn(model, CHECKPOINT_PATH)
            logger.info(
                f"  → Best model saved (val_exact_match={best_val_em:.4f})"
            )

    logger.info("-" * 90)
    logger.info(
        f"[ROUND 2 DONE] Best val_exact_match={best_val_em:.4f} "
        f"(round 1 baseline: {val0['val_exact_match']:.4f}, "
        f"improvement: {(best_val_em - val0['val_exact_match'])*100:+.2f}%)"
    )

    # Re-export ONNX
    if Path(CHECKPOINT_PATH).exists():
        try:
            best_model = load_crnn(CHECKPOINT_PATH, device="cpu")
            export_onnx(best_model, ONNX_PATH)
            logger.info(f"  ONNX re-exported: {ONNX_PATH}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-training round 2")
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help="Round 1 checkpoint to start from",
    )
    parser.add_argument(
        "--metadata", default="data/metadata.csv",
        help="Real data metadata CSV",
    )
    parser.add_argument("--image-dir", default="data")
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="Min confidence to keep prediction",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Round 2 không dùng synthetic regularizer",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error(
            f"Round 1 checkpoint không tồn tại: {args.checkpoint}\n"
            f"Hãy chạy `python train_crnn.py` trước."
        )
        return

    # ── Step 1: Predict + filter high confidence ────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1/2: Pseudo-label selection")
    logger.info("=" * 60)
    solver = CRNNCaptchaSolver(args.checkpoint)
    high_conf_csv, stats = select_high_confidence_samples(
        solver,
        Path(args.metadata),
        Path(args.image_dir),
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
    )
    del solver  # free memory

    if stats["high_confidence_correct"] < 50:
        logger.error(
            f"Chỉ có {stats['high_confidence_correct']} samples qua filter. "
            f"Round 1 model còn yếu, nên train round 1 thêm trước. "
            f"Hoặc giảm --confidence."
        )
        return

    # ── Step 2: Fine-tune ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2/2: Fine-tune trên high-confidence samples")
    logger.info("=" * 60)
    fine_tune_round2(
        model_path=args.checkpoint,
        high_conf_csv=high_conf_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_dir=Path(args.image_dir),
        use_synthetic=not args.no_synthetic,
    )

    logger.info("=" * 60)
    logger.info("Self-training complete. Run `python eval_crnn.py` to verify.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
