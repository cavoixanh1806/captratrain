"""
generate_trocr_synthetic.py
============================
Tao synthetic data cho TrOCR — co label biet truoc.

Khac voi generate_unet_data.py (sinh pair noisy+mask cho U-Net),
script nay sinh anh + label CSV de TrOCR co the train.

Output:
- data/synthetic/train/captcha_NNNNN.png + metadata.csv
- data/synthetic/val/captcha_NNNNN.png + metadata.csv

Usage:
    python generate_trocr_synthetic.py
"""

import csv
import logging
import random
from pathlib import Path

import cv2

from generate_unet_data import (
    render_text_on_image, random_text, CAPTCHA_SIZE,
    _load_real_backgrounds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path("data/synthetic")
TRAIN_COUNT = 5_000
VAL_COUNT = 1_000


def generate_split(output_dir: Path, count: int, split_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    # Xoa anh cu (giu lai cau truc thu muc)
    for f in output_dir.glob("captcha_*.png"):
        f.unlink()

    rows = []
    logger.info(f"Generating {count} samples for [{split_name}]...")

    for i in range(count):
        text = random_text()
        noisy, _ = render_text_on_image(text, CAPTCHA_SIZE)

        filename = f"captcha_{i:05d}.png"
        cv2.imwrite(str(output_dir / filename), noisy)
        rows.append({"filename": filename, "text": text})

        if (i + 1) % 500 == 0:
            logger.info(f"  [{split_name}] {i+1}/{count} done")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  [{split_name}] Done: {count} samples + metadata.csv")


def main():
    train_dir = OUTPUT_BASE / "train"
    val_dir = OUTPUT_BASE / "val"

    # Pre-load real backgrounds (dung chung cho ca train + val)
    _load_real_backgrounds()

    generate_split(train_dir, TRAIN_COUNT, "train")
    generate_split(val_dir, VAL_COUNT, "val")

    logger.info(f"[DONE] TrOCR synthetic data saved to: {OUTPUT_BASE}")
    logger.info(f"  Train: {TRAIN_COUNT} samples")
    logger.info(f"  Val:   {VAL_COUNT} samples")
    logger.info(f"")
    logger.info(f"Now you can train with:")
    logger.info(f"  python train.py --use-real-data --combine --augment")


if __name__ == "__main__":
    main()
