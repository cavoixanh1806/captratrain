"""
extract_real_backgrounds.py
============================
Tach background sach tu 500 anh CAPTCHA that bang inpainting MULTI-PASS.

Method detect text (4 layers):
1. HSV saturation > 30
2. Color ratio max/min > 1.35 — KEY: BG gray co R≈G≈B (ratio≈1), text co ratio cao
3. Color distance tu BG rim
4. Edge detection

Usage:
    python extract_real_backgrounds.py
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR: Path = Path("data")
OUTPUT_DIR: Path = Path("data/real_backgrounds")
INPUT_PATTERN: str = "map_*.png"


def detect_text_mask(img: np.ndarray) -> np.ndarray:
    """Detect text bang 4 methods ket hop.

    Color ratio la method chinh xac nhat cho CAPTCHA nay:
    - BG warm gray: R≈G≈B → max/min ≈ 1.0
    - Text do: R=200, B=50 → max/min = 4.0 → detect chinh xac
    - Text xanh: B=200, R=50 → max/min = 4.0 → detect chinh xac

    Args:
        img: BGR numpy array.

    Returns:
        Binary mask (uint8, 0/255), text=255.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    img_f = img.astype(np.float32)

    # Method 1: Saturation
    mask_sat = (s_channel > 30).astype(np.uint8) * 255

    # Method 2: Color ratio max/min — CHINH XAC NHAT
    ch_max = img_f.max(axis=2)
    ch_min = img_f.min(axis=2)
    color_ratio = ch_max / (ch_min + 1.0)
    mask_ratio = (color_ratio > 1.35).astype(np.uint8) * 255

    # Method 3: Color distance tu BG (rim pixels)
    border = 5
    rim = np.concatenate([
        img[:border, :].reshape(-1, 3),
        img[-border:, :].reshape(-1, 3),
        img[:, :border].reshape(-1, 3),
        img[:, -border:].reshape(-1, 3),
    ])
    bg_color = np.median(rim, axis=0).astype(np.float32)
    color_diff = np.sqrt(((img_f - bg_color) ** 2).sum(axis=2))
    mask_dist = (color_diff > 35).astype(np.uint8) * 255

    # Method 4: Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_edge = cv2.dilate(edges, edge_k, iterations=2)

    # Union tat ca
    mask = cv2.bitwise_or(mask_sat, mask_ratio)
    mask = cv2.bitwise_or(mask, mask_dist)
    mask = cv2.bitwise_or(mask, mask_edge)

    # Fill holes + dilate aggressive
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, dilate_k, iterations=2)

    return mask


def inpaint_multipass(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpainting 3 pass de xoa triet de text.

    Pass 1: TELEA (nhanh)
    Pass 2: NS (smooth)
    Pass 3: Soft blur blend tren vung mask
    """
    result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    big_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_big = cv2.dilate(mask, big_k, iterations=1)
    result = cv2.inpaint(result, mask_big, 5, cv2.INPAINT_NS)

    blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
    alpha = mask_big.astype(np.float32) / 255.0
    alpha_3ch = cv2.GaussianBlur(
        np.stack([alpha] * 3, axis=-1), (11, 11), 3.0
    )
    result = result.astype(np.float32) * (1 - alpha_3ch) + \
             blurred.astype(np.float32) * alpha_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def is_clean(img: np.ndarray, threshold: int = 60) -> bool:
    """Kiem tra anh da sach (p99 saturation < threshold)."""
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.percentile(s, 99)) < threshold


def process_one_image(input_path: Path, output_path: Path) -> bool:
    """Xu ly 1 anh: detect → inpaint → save.

    Luu y: Tat ca anh deu duoc giu lai (ke ca anh con ghost nhe).
    Ly do: U-Net se hoc ignore ca ghost chu lan chu moi render len.
    Chi skip anh bi loi doc file.

    Returns:
        True neu doc va xu ly duoc, False neu loi file.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        return False

    mask = detect_text_mask(img)
    cleaned = inpaint_multipass(img, mask)

    # Retry voi mask lon hon neu chua sach
    if not is_clean(cleaned):
        extra_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_extra = cv2.dilate(mask, extra_k, iterations=2)
        cleaned = cv2.inpaint(cleaned, mask_extra, 9, cv2.INPAINT_TELEA)
        cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0.8)

    cv2.imwrite(str(output_path), cleaned)
    return True  # Giu lai tat ca, ke ca anh con ghost nhe


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(INPUT_DIR.glob(INPUT_PATTERN))
    if not images:
        logger.error(f"Khong tim thay anh nao trong {INPUT_DIR}/{INPUT_PATTERN}")
        return

    logger.info(f"Tim thay {len(images)} anh. Bat dau extract backgrounds...")

    success = 0
    failed = 0
    for i, img_path in enumerate(images):
        out_path = OUTPUT_DIR / f"bg_{img_path.stem}.png"
        if process_one_image(img_path, out_path):
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(images)} — processed={success}, failed={failed}")

    logger.info(f"[DONE] Processed {success}/{len(images)} backgrounds")
    logger.info(f"Note: Tat ca BG duoc giu lai (ke ca ghost nhe) — U-Net se hoc ignore.")


if __name__ == "__main__":
    main()
