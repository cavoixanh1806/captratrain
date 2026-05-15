"""
dataset_crnn.py
================
Dataset wrapper cho CRNN CAPTCHA training.

Hỗ trợ 2 nguồn:
    - Real data (data/, metadata.csv)
    - Synthetic data (data/synthetic_crnn/, metadata.csv hoặc filename = label)

Augmentation cho CRNN khác với TrOCR — vì đầu vào là raw RGB, không qua U-Net,
augmentation phải mô phỏng đủ noise của Minecraft map captcha:
    - ColorJitter: brightness/contrast/saturation/hue
    - GaussNoise nhẹ
    - GaussianBlur ngẫu nhiên
    - Affine xoay nhẹ + translate (±5%)
    - Cutout (random erasing) nhẹ — robust với noise

KHÔNG dùng ElasticTransform/GridDistortion vì có thể bẻ shape ký tự.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from crnn_model import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    encode_label,
)

try:
    import albumentations as A
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False


# Mean/std cho normalize — ImageNet stats là an toàn cho RGB tự nhiên
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _build_albu_aug(strong: bool = True) -> "A.Compose | None":
    """Build albumentations augmentation pipeline cho train.

    Args:
        strong: True để bật augment mạnh hơn (cho train với it data).

    Returns:
        A.Compose hoặc None nếu albumentations không có.
    """
    if not _HAS_ALBU:
        return None
    if strong:
        return A.Compose([
            A.Affine(
                rotate=(-12, 12),
                translate_percent=(-0.06, 0.06),
                scale=(0.85, 1.15),
                shear=(-5, 5),
                p=0.6,
                mode=cv2.BORDER_REFLECT_101,
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15,
                p=0.4,
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=3),
            ], p=0.25),
            A.CoarseDropout(
                max_holes=4, max_height=8, max_width=8,
                min_holes=1, min_height=4, min_width=4,
                fill_value=0, p=0.2,
            ),
        ])
    else:
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3,
            ),
            A.GaussNoise(var_limit=(3.0, 10.0), p=0.2),
        ])


_TRAIN_AUG = _build_albu_aug(strong=True) if _HAS_ALBU else None

# Torchvision fallback nếu không có albumentations
_TV_TRAIN_AUG = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.85, 1.15), shear=5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
])


def _resize_and_normalize(rgb: np.ndarray) -> torch.Tensor:
    """Resize → CHW float [0,1] → normalize.

    Args:
        rgb: (H, W, 3) uint8 RGB array.

    Returns:
        Tensor (3, INPUT_HEIGHT, INPUT_WIDTH) normalized.
    """
    resized = cv2.resize(
        rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR,
    )
    arr = resized.astype(np.float32) / 255.0
    # Normalize per channel
    for i, (m, s) in enumerate(zip(_MEAN, _STD)):
        arr[..., i] = (arr[..., i] - m) / s
    # HWC → CHW
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class CRNNCaptchaDataset(Dataset):
    """Dataset CRNN cho CAPTCHA.

    Args:
        image_dir: thư mục chứa ảnh.
        metadata_path: path đến metadata.csv (cột filename, text).
        augment: True để bật augmentation cho train.
    """

    def __init__(
        self,
        image_dir: str | Path,
        metadata_path: str | Path,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.augment = augment

        df = pd.read_csv(metadata_path, dtype=str)
        df = df.dropna(subset=["filename", "text"]).reset_index(drop=True)
        df["text"] = df["text"].str.strip().str.upper()
        df = df[df["text"].str.len() == 5].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        filename: str = row["filename"]
        text: str = row["text"]

        img_path = self.image_dir / filename
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Augmentation (chỉ khi train)
        if self.augment:
            if _TRAIN_AUG is not None:
                rgb = _TRAIN_AUG(image=rgb)["image"]
            else:
                pil = Image.fromarray(rgb)
                pil = _TV_TRAIN_AUG(pil)
                rgb = np.array(pil)

        pixel_values = _resize_and_normalize(rgb)

        label_ids = torch.tensor(encode_label(text), dtype=torch.long)
        label_len = len(text)

        return pixel_values, label_ids, label_len, text


def collate_fn(batch: list) -> dict:
    """Custom collate cho CTC: ghép pixel_values, labels (concat), label_lengths.

    Trả về dict để Trainer tự nhận diện.
    """
    pixels = torch.stack([b[0] for b in batch], dim=0)            # (B, 3, H, W)
    labels = torch.cat([b[1] for b in batch], dim=0)               # sum-len
    label_lengths = torch.tensor(
        [b[2] for b in batch], dtype=torch.long,
    )                                                              # (B,)
    texts = [b[3] for b in batch]
    return {
        "images": pixels,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": texts,
    }


def create_crnn_datasets(
    real_data_dir: str | Path = "data",
    synthetic_dir: str | Path = "data/synthetic_crnn",
    val_split: float = 0.15,
    use_real: bool = True,
    use_synthetic: bool = True,
    augment_train: bool = True,
    seed: int = 42,
) -> tuple[CRNNCaptchaDataset, CRNNCaptchaDataset]:
    """Build train/val datasets — kết hợp real + synthetic theo flag.

    Strategy:
        - Real: chia train/val theo `val_split`.
        - Synthetic: 100% train (val luôn từ real để eval đúng domain).

    Args:
        real_data_dir: thư mục data/ (real CAPTCHA + metadata.csv).
        synthetic_dir: thư mục data/synthetic_crnn/ (synthetic + metadata.csv).
        val_split: tỷ lệ real → val.
        use_real: True để bao gồm real data.
        use_synthetic: True để bao gồm synthetic data.
        augment_train: True để augment trên tập train.
        seed: random seed cho train/val split.

    Returns:
        Tuple (train_dataset, val_dataset). Train có thể là ConcatDataset
        nếu kết hợp real+synthetic. Val luôn là real.
    """
    from torch.utils.data import ConcatDataset

    real_data_dir = Path(real_data_dir)
    synthetic_dir = Path(synthetic_dir)

    train_parts: list[Dataset] = []
    val_dataset: Dataset | None = None

    # ── Real data ─────────────────────────────────────────────────────────────
    if use_real:
        meta_real = real_data_dir / "metadata.csv"
        if not meta_real.exists():
            raise FileNotFoundError(
                f"Real metadata not found: {meta_real}. "
                f"Run label_server.py + import_new_data.py first."
            )

        df_all = pd.read_csv(meta_real, dtype=str).dropna()
        df_all["text"] = df_all["text"].str.strip().str.upper()
        df_all = df_all[df_all["text"].str.len() == 5].reset_index(drop=True)

        n_val = max(1, int(len(df_all) * val_split))
        df_val = df_all.sample(n=n_val, random_state=seed)
        df_train_real = df_all.drop(df_val.index)

        # Save temp metadata for sub-datasets
        train_meta = real_data_dir / "_crnn_train.csv"
        val_meta = real_data_dir / "_crnn_val.csv"
        df_train_real.to_csv(train_meta, index=False)
        df_val.to_csv(val_meta, index=False)

        train_parts.append(CRNNCaptchaDataset(
            real_data_dir, train_meta, augment=augment_train,
        ))
        val_dataset = CRNNCaptchaDataset(
            real_data_dir, val_meta, augment=False,
        )

        # Cleanup temp files (datasets đã đọc xong vào memory tại __init__)
        train_meta.unlink(missing_ok=True)
        val_meta.unlink(missing_ok=True)

    # ── Synthetic data ────────────────────────────────────────────────────────
    if use_synthetic:
        meta_syn = synthetic_dir / "metadata.csv"
        if not meta_syn.exists():
            raise FileNotFoundError(
                f"Synthetic metadata not found: {meta_syn}. "
                f"Run generate_synthetic_crnn.py first."
            )
        train_parts.append(CRNNCaptchaDataset(
            synthetic_dir, meta_syn, augment=augment_train,
        ))

    if not train_parts:
        raise ValueError("Phải bật ít nhất một trong use_real / use_synthetic")

    if val_dataset is None:
        # Trường hợp chỉ synthetic (không khuyến nghị) — split synthetic
        full = train_parts[0]
        n_val = max(1, int(len(full) * val_split))
        n_train = len(full) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        return train_dataset, val_dataset

    train_dataset = (
        train_parts[0] if len(train_parts) == 1
        else ConcatDataset(train_parts)
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Smoke test với real data
    ds = CRNNCaptchaDataset(
        image_dir="data",
        metadata_path="data/metadata.csv",
        augment=True,
    )
    print(f"Dataset size: {len(ds)}")
    pv, lbl, ln, text = ds[0]
    print(f"Sample 0: text={text!r}, label_ids={lbl.tolist()}, len={ln}")
    print(f"  pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
    print(f"  pixel range: [{pv.min().item():.3f}, {pv.max().item():.3f}]")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"  images: {batch['images'].shape}")
    print(f"  labels (concat): {batch['labels'].shape}")
    print(f"  label_lengths: {batch['label_lengths'].tolist()}")
    print(f"  texts: {batch['texts']}")
