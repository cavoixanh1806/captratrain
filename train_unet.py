"""
train_unet.py
=============
Train the U-Net denoiser for CAPTCHA text segmentation.

The U-Net learns to predict a per-pixel probability map:
    - 0 = background (noise, lines, gradient)
    - 1 = text character pixel

Training uses combined Dice + BCE loss:
    - BCE: chuẩn binary classification per-pixel
    - Dice: focus vào overlap, xử lý class imbalance tốt hơn
      (vì text chỉ chiếm ~15-20% diện tích → BCE một mình bias về background)

Usage:
    python train_unet.py
    python train_unet.py --epochs 50 --batch-size 16
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from unet_model import CaptchaUNet


class DiceBCELoss(nn.Module):
    """Combined Dice Loss + BCE Loss.

    Dice Loss tập trung vào overlap (IoU-like), không phụ thuộc class balance.
    BCE Loss đảm bảo độ chính xác pixel-level.
    Kết hợp cả hai cho kết quả ổn định nhất với imbalanced segmentation.

    Args:
        dice_weight: Trọng số cho Dice Loss (mặc định 0.5).
        bce_weight: Trọng số cho BCE Loss (mặc định 0.5).
        smooth: Hằng số tránh chia 0.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Tính loss.

        Args:
            pred: Predicted probability map (B, 1, H, W), values 0-1.
            target: Ground truth binary mask (B, 1, H, W), values 0 or 1.

        Returns:
            Combined loss (scalar).
        """
        # BCE Loss — clamp pred để tránh log(0)
        pred_clamped = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = F.binary_cross_entropy(pred_clamped, target)

        # Dice Loss — flatten tensors rồi tính intersection/union
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return self.bce_weight * bce + self.dice_weight * dice



# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR: str = "data/unet_pairs"
MODEL_SAVE_PATH: str = "captcha_unet_model.pth"
BATCH_SIZE: int = 32       # 128x128 images, RTX 3060 handles 32 easily
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30
IMAGE_SIZE: int = 128


class UNetPairDataset(Dataset):
    """Dataset for U-Net training: paired noisy images and binary masks.

    Each sample is a pair:
        - Input: Noisy CAPTCHA image (3, 128, 128), normalized to [0, 1]
        - Target: Binary mask (1, 128, 128), values 0 or 1

    Args:
        data_dir: Directory containing 'noisy/' and 'mask/' subdirectories.
        augment: Whether to apply random augmentations during training.
    """

    def __init__(self, data_dir: str | Path, augment: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.noisy_dir = self.data_dir / "noisy"
        self.mask_dir = self.data_dir / "mask"
        self.augment = augment

        # List all paired files
        self.filenames = sorted([
            f.name for f in self.noisy_dir.glob("*.png")
            if (self.mask_dir / f.name).exists()
        ])

        if not self.filenames:
            raise FileNotFoundError(
                f"No paired images found in {self.noisy_dir} and {self.mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        filename = self.filenames[idx]

        # Load noisy image (BGR -> RGB -> normalize)
        noisy = cv2.imread(str(self.noisy_dir / filename))
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale -> binary)
        mask = cv2.imread(str(self.mask_dir / filename), cv2.IMREAD_GRAYSCALE)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                noisy = np.fliplr(noisy).copy()
                mask = np.fliplr(mask).copy()

            # Random brightness/contrast
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-15, 15)     # brightness
            noisy = np.clip(noisy * alpha + beta, 0, 255).astype(np.uint8)

        # Normalize to [0, 1] and convert to tensors
        noisy_tensor = torch.from_numpy(noisy).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return {
            "image": noisy_tensor,    # (3, 128, 128)
            "mask": mask_tensor,      # (1, 128, 128)
        }


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU (Intersection over Union) for binary segmentation.

    Args:
        pred: Predicted probability map (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Threshold to binarize predictions.

    Returns:
        IoU score (0-1).
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = ((pred_binary + target) > 0).float().sum()

    if union == 0:
        return 1.0  # Both empty
    return (intersection / union).item()


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute pixel-level accuracy.

    Args:
        pred: Predicted probability map.
        target: Ground truth mask.
        threshold: Threshold for binarization.

    Returns:
        Accuracy (0-1).
    """
    pred_binary = (pred > threshold).float()
    correct = (pred_binary == target).float().sum()
    total = target.numel()
    return (correct / total).item()


def train_one_epoch(
    model: CaptchaUNet,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict with 'loss', 'iou', 'pixel_acc'.
    """
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            total_loss += loss.item()
            total_iou += compute_iou(predictions, masks)
            total_acc += compute_pixel_accuracy(predictions, masks)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "pixel_acc": total_acc / num_batches,
    }


@torch.no_grad()
def validate(
    model: CaptchaUNet,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate model.

    Returns:
        Dict with 'loss', 'iou', 'pixel_acc'.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        predictions = model(images)
        loss = criterion(predictions, masks)

        total_loss += loss.item()
        total_iou += compute_iou(predictions, masks)
        total_acc += compute_pixel_accuracy(predictions, masks)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "pixel_acc": total_acc / num_batches,
    }


def main(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> None:
    """Main training loop."""
    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dir = Path(DATA_DIR) / "train"
    val_dir = Path(DATA_DIR) / "val"

    if not train_dir.exists():
        logger.error(f"Training data not found: {train_dir}")
        logger.error("Run 'python generate_unet_data.py' first!")
        return

    train_dataset = UNetPairDataset(train_dir, augment=True)
    val_dataset = UNetPairDataset(val_dir, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CaptchaUNet().to(device)
    logger.info(f"Model params: {model.count_parameters():,}")

    # ── Training setup ────────────────────────────────────────────────────────
    # BCELoss + DiceLoss kết hợp — xử lý class imbalance tốt hơn BCE một mình
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_iou = 0.0
    best_epoch = 0

    logger.info(f"Training for {epochs} epochs...")
    logger.info("-" * 80)

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t_start

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
            f"Acc: {val_metrics['pixel_acc']:.4f}"
        )

        # Save best model
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"  -> Best model saved (IoU: {best_val_iou:.4f})")

    logger.info("-" * 80)
    logger.info(f"[DONE] Best val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net CAPTCHA denoiser")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
