"""
train_crnn.py
==============
Train CRNN+CTC cho Minecraft Map CAPTCHA (pipeline tối giản).

Tạm thời KHÔNG dùng:
    - EMA weights
    - EarlyStopping
    - Self-training (chạy riêng `python self_train.py` nếu cần)
    - Synthetic data (chạy `python generate_synthetic_crnn.py` riêng nếu cần)

Pipeline hiện tại:
    1. Load CRNN (~2.18M params)
    2. Loss: CTCLoss + zero_infinity=True
    3. Optimizer: AdamW lr=1e-3, weight_decay=1e-4
    4. LR schedule: linear warmup 200 steps → cosine decay (per-batch step)
    5. Mixed-precision (fp16 autocast) trên CUDA
    6. Train trên real data (754 ảnh × 0.85), eval trên val (754 × 0.15)
    7. Augmentation theo research doc: Affine + ColorJitter + Noise + Blur + Cutout
    8. Train hết epochs (không early stop) — save best checkpoint theo val_em
    9. ONNX export cuối cùng

Usage:
    python train_crnn.py                         # default: 200 epochs, real only
    python train_crnn.py --epochs 80
    python train_crnn.py --use-synthetic         # bật synthetic (cần generate trước)
    python train_crnn.py --resume                # resume from last
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crnn_model import (
    CRNN,
    CTC_BLANK_INDEX,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
    decode_greedy,
    save_crnn,
    export_onnx,
    load_crnn,
)
from dataset_crnn import collate_fn, create_crnn_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH: str = "captcha_crnn_model.pth"
ONNX_PATH: str = "captcha_crnn_model.onnx"
LAST_CHECKPOINT_PATH: str = "captcha_crnn_last.pth"

DEFAULT_EPOCHS: int = 200
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 1e-3
WARMUP_STEPS: int = 200
GRAD_CLIP_NORM: float = 5.0


# ─── Scheduler ───────────────────────────────────────────────────────────────


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR schedule: linear warmup → cosine decay.

    - Steps 0 → warmup_steps:   LR ramp 0 → base_lr
    - Steps warmup → total:     LR cosine decay từ base_lr → base_lr * min_lr_ratio
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(
    preds: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Tính exact_match + CER."""
    if not preds:
        return {"exact_match": 0.0, "cer": 1.0}

    exact = sum(p == l for p, l in zip(preds, labels)) / len(preds)

    total_dist = 0
    total_chars = 0
    for p, l in zip(preds, labels):
        total_dist += _edit_distance(p, l)
        total_chars += max(len(l), 1)
    cer = total_dist / max(total_chars, 1)

    return {"exact_match": exact, "cer": cer}


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (c1 != c2)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


# ─── Train / Validate ────────────────────────────────────────────────────────


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: "torch.cuda.amp.GradScaler | None",
    device: torch.device,
    epoch_idx: int,
    log_interval: int = 50,
) -> dict[str, float]:
    """Train 1 epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[str] = []
    all_labels: list[str] = []

    from tqdm import tqdm
    pbar = tqdm(loader, leave=False, dynamic_ncols=True, ascii=False)
    for step, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)
        texts: list[str] = batch["texts"]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1)
                T_size = log_probs.size(0)
                B_size = log_probs.size(1)
                input_lengths = torch.full(
                    (B_size,), T_size, dtype=torch.long, device=device,
                )
                loss = criterion(
                    log_probs, labels, input_lengths, label_lengths,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            T_size = log_probs.size(0)
            B_size = log_probs.size(1)
            input_lengths = torch.full(
                (B_size,), T_size, dtype=torch.long, device=device,
            )
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        # Step scheduler MỖI BATCH (warmup + cosine theo batch step)
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        # Decode every N steps để khỏi chậm — chỉ sample
        if step % log_interval == 0:
            with torch.no_grad():
                preds = decode_greedy(logits.detach())
            all_preds.extend(preds[:8])
            all_labels.extend(texts[:8])

    avg_loss = total_loss / max(n_batches, 1)
    
    # HF style log at end of epoch
    lr_now = optimizer.param_groups[0]['lr']
    gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    log_dict = {
        'loss': f"{avg_loss:.4f}",
        'grad_norm': f"{gn:.3f}",
        'learning_rate': f"{lr_now:.3e}",
        'epoch': f"{epoch_idx}"
    }
    tqdm.write(str(log_dict).replace('"', "'"))
    
    metrics = compute_metrics(all_preds, all_labels)
    return {
        "loss": avg_loss,
        **{f"train_{k}": v for k, v in metrics.items()},
    }


@torch.no_grad()
def validate(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> dict[str, float]:
    """Validate trên toàn bộ val set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[str] = []
    all_labels: list[str] = []

    from tqdm import tqdm
    for batch in tqdm(loader, leave=False, dynamic_ncols=True, ascii=False):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)
        texts: list[str] = batch["texts"]

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=-1)
        T_size = log_probs.size(0)
        B_size = log_probs.size(1)
        input_lengths = torch.full(
            (B_size,), T_size, dtype=torch.long, device=device,
        )
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item()
        n_batches += 1

        preds = decode_greedy(logits)
        all_preds.extend(preds)
        all_labels.extend(texts)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_preds, all_labels)
    return {
        "loss": avg_loss,
        **{f"val_{k}": v for k, v in metrics.items()},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    use_synthetic: bool = False,
    use_real: bool = True,
    augment: bool = True,
    resume: bool = False,
) -> None:
    """Main training loop."""
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed-precision (fp16) enabled")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = create_crnn_datasets(
        use_real=use_real,
        use_synthetic=use_synthetic,
        augment_train=augment,
    )
    logger.info(
        f"Train: {len(train_ds):,} samples (synthetic={'on' if use_synthetic else 'off'})"
    )
    logger.info(f"Val (real only): {len(val_ds):,} samples")

    num_workers = 0  # Windows: 0 để tránh pickle issue với albumentations
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    logger.info(f"CRNN params: {model.count_parameters():,}")

    criterion = nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps // 10),
        total_steps=total_steps,
    )
    logger.info(
        f"LR schedule: warmup={min(WARMUP_STEPS, total_steps // 10)} steps, "
        f"total={total_steps} steps ({steps_per_epoch} steps/epoch × {epochs} epochs)"
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_em = 0.0
    best_epoch = 0
    if resume and Path(LAST_CHECKPOINT_PATH).exists():
        ckpt = torch.load(LAST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_em = ckpt.get("best_val_em", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        logger.info(
            f"Resumed from epoch {ckpt['epoch']} "
            f"(best_val_exact_match={best_val_em:.4f} @ epoch {best_epoch})"
        )

    # ── Train loop (KHÔNG early stop, chạy hết epochs) ────────────────────────
    logger.info(f"Training for {epochs} epochs (start at {start_epoch})...")
    logger.info("-" * 90)

    for epoch in range(start_epoch, epochs + 1):
        t_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch_idx=epoch,
        )
        
        t_eval_start = time.time()
        val_metrics = validate(model, val_loader, criterion, device)
        eval_runtime = time.time() - t_eval_start
        
        eval_dict = {
            'eval_loss': f"{val_metrics['loss']:.3f}",
            'eval_cer': f"{val_metrics['val_cer']:.3f}",
            'eval_exact_match': f"{val_metrics['val_exact_match']:.3f}",
            'eval_runtime': f"{eval_runtime:.2f}",
            'eval_samples_per_second': f"{len(val_ds) / max(eval_runtime, 1e-5):.3f}",
            'eval_steps_per_second': f"{len(val_loader) / max(eval_runtime, 1e-5):.3f}",
            'epoch': str(epoch)
        }
        print(str(eval_dict).replace('"', "'"))

        # Save last (cho resume)
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val_em": best_val_em,
            "best_epoch": best_epoch,
        }, LAST_CHECKPOINT_PATH)

        # Save best (theo val exact_match)
        if val_metrics["val_exact_match"] > best_val_em:
            best_val_em = val_metrics["val_exact_match"]
            best_epoch = epoch
            save_crnn(model, CHECKPOINT_PATH)
            logger.info(
                f"  → Best model saved "
                f"(val_exact_match={best_val_em:.4f}, val_cer={val_metrics['val_cer']:.4f})"
            )

    logger.info("-" * 90)
    logger.info(
        f"[DONE] Best val_exact_match={best_val_em:.4f} at epoch {best_epoch}"
    )
    logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    if Path(CHECKPOINT_PATH).exists():
        try:
            best_model = load_crnn(CHECKPOINT_PATH, device="cpu")
            export_onnx(best_model, ONNX_PATH)
            logger.info(f"  ONNX exported: {ONNX_PATH}")
        except Exception as e:
            logger.warning(f"ONNX export failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN+CTC for CAPTCHA")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--use-synthetic", action="store_true",
        help="Bật synthetic data (cần generate trước với generate_synthetic_crnn.py)",
    )
    parser.add_argument(
        "--no-real", action="store_true",
        help="Disable real data (chỉ synthetic — KHÔNG khuyến nghị)",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable augmentation",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_synthetic=args.use_synthetic,
        use_real=not args.no_real,
        augment=not args.no_augment,
        resume=args.resume,
    )
