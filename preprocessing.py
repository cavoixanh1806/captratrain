"""
preprocessing.py
=================
Module tien xu ly anh CAPTCHA truoc khi dua vao TrOCR.

Chi dung 1 phuong phap duy nhat:
- unet: U-Net denoiser — soft probability map (tot nhat, khong phu thuoc mau sac)

Pipeline: CAPTCHA -> U-Net -> soft grayscale -> TrOCR -> Text
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ── Cached U-Net model (singleton) ────────────────────────────────────────────
_unet_model = None
_unet_device = None

UNET_MODEL_PATH = "captcha_unet_model.pth"


def _get_unet_model():
    """Load U-Net model once and cache it."""
    global _unet_model, _unet_device
    if _unet_model is None:
        from unet_model import CaptchaUNet
        _unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _unet_model = CaptchaUNet()
        _unet_model.load_state_dict(
            torch.load(UNET_MODEL_PATH, map_location=_unet_device)
        )
        _unet_model.to(_unet_device)
        _unet_model.eval()
    return _unet_model, _unet_device


def preprocess_captcha(image: np.ndarray) -> Image.Image:
    """Tien xu ly CAPTCHA bang U-Net denoiser.

    Dung soft probability map truc tiep lam grayscale thay vi binary threshold.
    - prob=1.0 (text pixel) -> intensity=0 (den)
    - prob=0.0 (background) -> intensity=255 (trang)
    Ket qua: anh grayscale sach, chu den tren nen trang, giu soft edges.
    Khong phu thuoc mau sac cua ky tu -> xu ly dung ca khi mau ky tu trung nhau.

    Args:
        image: BGR numpy array (128, 128, 3).

    Returns:
        PIL RGB Image da tien xu ly.
    """
    model, device = _get_unet_model()

    # Convert BGR -> RGB, normalize to [0, 1], add batch dim
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 128, 128)

    # Predict pixel probabilities
    with torch.no_grad():
        prob_map = model(tensor)  # (1, 1, 128, 128), values 0-1

    # Dung soft probability map truc tiep — khong binarize
    # Invert: text (prob~1) -> dark (0), background (prob~0) -> white (255)
    prob_np = prob_map.squeeze().cpu().numpy()  # (128, 128), float 0-1
    clean_gray = ((1.0 - prob_np) * 255.0).astype(np.uint8)

    # Stack thanh RGB (3 kenh) de TrOCR processor nhan dung format
    clean_rgb = np.stack([clean_gray, clean_gray, clean_gray], axis=-1)
    return Image.fromarray(clean_rgb)


def preprocess_from_path(image_path: str | Path) -> Image.Image:
    """Tien ich: doc anh tu path va preprocess.

    Args:
        image_path: Duong dan anh CAPTCHA.

    Returns:
        Anh PIL da tien xu ly.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Khong tim thay: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    return preprocess_captcha(img)
