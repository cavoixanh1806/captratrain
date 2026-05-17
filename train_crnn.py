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
    3. Optimizer: AdamW lr=5e-4 (DEFAULT_LR), weight_decay=1e-4
    4. LR schedule: linear warmup (>=2 epochs, floor WARMUP_STEPS=200) → cosine decay (per-batch step)
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
import csv
import logging
import math
import os
import time
from datetime import datetime, timezone
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
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 3e-4
WARMUP_STEPS: int = 200
GRAD_CLIP_NORM: float = 5.0


# ─── DataLoader worker auto-policy ───────────────────────────────────────────


def _auto_num_workers() -> int:
    """Choose a sensible default ``num_workers`` for the host.

    Policy (crnn-ctc-collapse-fix design §B item 3):

    - Windows (``os.name == "nt"``):
        * If albumentations imports cleanly, use ``min(4, cpu_count // 2)``.
          ``_TRAIN_AUG`` is a module-level ``albumentations.Compose`` and is
          pickleable under the spawn start method (verified with
          albumentations 1.4.3).
        * If albumentations import fails (torchvision fallback path), drop
          to ``0`` to avoid pickling the torchvision ``Compose`` across
          spawn workers.
    - Linux/macOS: ``min(8, cpu_count // 2)``.

    Returns 0 whenever ``cpu_count`` is unavailable or computes to <= 0,
    so the DataLoader keeps the main-thread fallback that has always
    worked.
    """
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0

    if os.name == "nt":
        try:
            import albumentations  # noqa: F401  (probe only)
        except Exception:
            return 0
        return max(0, min(4, cpu_count // 2))

    return max(0, min(8, cpu_count // 2))


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
        "grad_norm": float(gn),
        "learning_rate": float(lr_now),
        **{f"train_{k}": v for k, v in metrics.items()},
    }


# ─── Metrics CSV writer ──────────────────────────────────────────────────────


METRICS_CSV_FIELDS: tuple[str, ...] = (
    "epoch",
    "timestamp",
    "train_loss",
    "train_grad_norm",
    "learning_rate",
    "eval_loss",
    "eval_cer",
    "eval_exact_match",
    "eval_runtime_s",
    "gap",
    "is_best",
    "best_val_em",
    "best_epoch",
)


def _open_metrics_csv(path: Path, resume: bool) -> "tuple[object, csv.DictWriter] | tuple[None, None]":
    """Open metrics CSV file for appending, creating header if missing.

    When ``resume=True`` and the file already exists, append without header.
    Otherwise truncate and write a fresh header.
    """
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume and path.exists() and path.stat().st_size > 0)
    mode = "a" if resume and path.exists() else "w"
    f = path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()
    return f, writer


def _append_metrics_row(
    writer: "csv.DictWriter | None",
    f: "object | None",
    row: dict,
) -> None:
    if writer is None or f is None:
        return
    # Ensure all keys are present, fill missing with empty string.
    safe_row = {k: row.get(k, "") for k in METRICS_CSV_FIELDS}
    writer.writerow(safe_row)
    try:
        f.flush()
    except Exception:
        pass


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
    *,
    num_workers: int | None = None,
    metrics_csv: "str | os.PathLike | None" = None,
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

    # DataLoader: pick num_workers via the auto-policy when the caller didn't
    # override. Albumentations 1.4.3 + module-level `_TRAIN_AUG` Compose pickle
    # correctly under Windows spawn, so multi-worker is safe on the target
    # hardware. Fallbacks (torchvision aug, low-core hosts, explicit override)
    # all collapse to num_workers=0, which preserves the legacy behaviour.
    if num_workers is None:
        num_workers = _auto_num_workers()
    num_workers = max(0, int(num_workers))

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_amp,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(
        f"DataLoader: num_workers={num_workers}, "
        f"persistent_workers={num_workers > 0}, "
        f"prefetch_factor={4 if num_workers > 0 else 'n/a'}, "
        f"pin_memory={use_amp}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    logger.info(f"CRNN params: {model.count_parameters():,}")

    criterion = nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    # Warmup: at least WARMUP_STEPS (floor for tiny datasets) AND at least
    # 2 full epochs of warmup so that LR doesn't ramp to peak before the
    # network has seen enough batches (crnn-ctc-collapse-fix design §B item 2).
    warmup_steps = max(WARMUP_STEPS, steps_per_epoch * 2)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    logger.info(
        f"LR schedule: warmup={warmup_steps} steps, "
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

    # ── Metrics CSV (per-epoch table for plotting / spreadsheets) ────────────
    metrics_csv_path = Path(metrics_csv) if metrics_csv else None
    csv_file, csv_writer = _open_metrics_csv(metrics_csv_path, resume=resume)
    if metrics_csv_path is not None:
        logger.info(f"Metrics CSV: {metrics_csv_path}")

    # ── Train loop (KHÔNG early stop, chạy hết epochs) ────────────────────────
    logger.info(f"Training for {epochs} epochs (start at {start_epoch})...")
    logger.info("-" * 90)

    try:
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

            # Save best (theo val exact_match). LUÔN save ở epoch 1 để có
            # checkpoint baseline (tránh case `best_val_em == 0` suốt training
            # khiến file best.pth không bao giờ được tạo và eval crash).
            is_best = (
                val_metrics["val_exact_match"] > best_val_em
                or epoch == start_epoch
            )
            if is_best:
                best_val_em = val_metrics["val_exact_match"]
                best_epoch = epoch
                save_crnn(model, CHECKPOINT_PATH)
                logger.info(
                    f"  → Best model saved "
                    f"(val_exact_match={best_val_em:.4f}, val_cer={val_metrics['val_cer']:.4f})"
                )

            # Append metrics row (after best-tracker so is_best is correct).
            _append_metrics_row(
                csv_writer, csv_file,
                {
                    "epoch": epoch,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "train_loss": f"{train_metrics['loss']:.6f}",
                    "train_grad_norm": f"{train_metrics['grad_norm']:.6f}",
                    "learning_rate": f"{train_metrics['learning_rate']:.6e}",
                    "eval_loss": f"{val_metrics['loss']:.6f}",
                    "eval_cer": f"{val_metrics['val_cer']:.6f}",
                    "eval_exact_match": f"{val_metrics['val_exact_match']:.6f}",
                    "eval_runtime_s": f"{eval_runtime:.3f}",
                    "gap": f"{val_metrics['loss'] - train_metrics['loss']:.6f}",
                    "is_best": "1" if is_best else "0",
                    "best_val_em": f"{best_val_em:.6f}",
                    "best_epoch": best_epoch,
                },
            )
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

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
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help=(
            "DataLoader worker count. Default (None) picks an auto policy: "
            "Windows → min(4, cpu_count//2) (or 0 if albumentations import "
            "fails); Linux/macOS → min(8, cpu_count//2). Pass 0 to force "
            "the legacy main-thread loader."
        ),
    )
    parser.add_argument(
        "--metrics-csv", default=None,
        help=(
            "Optional path to write a per-epoch metrics CSV "
            "(epoch, train_loss, eval_loss, eval_exact_match, gap, ...). "
            "Header is written once; rows are appended each epoch and on "
            "--resume new rows continue the existing file."
        ),
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
        num_workers=args.num_workers,
        metrics_csv=args.metrics_csv,
    )
