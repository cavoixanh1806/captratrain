"""
generate_synthetic_crnn.py
===========================
Sinh synthetic CAPTCHA cho CRNN training (theo research doc).

Tái sử dụng `render_text_on_image` từ generate_unet_data.py (đã calibrated từ
754 ảnh real Minecraft map: BGR avg, saturation, texture, font, color, overlap).

Output:
    data/synthetic_crnn/captcha_NNNNNN.png — 128x128 RGB
    data/synthetic_crnn/metadata.csv       — filename,text

Theo research doc: dataset 50K-200K. Mặc định 50K (cân bằng giữa thời gian
generate + RAM + chất lượng). Có thể tăng bằng --count.

Usage:
    python generate_synthetic_crnn.py            # 50K samples
    python generate_synthetic_crnn.py --count 100000
"""

import argparse
import csv
import logging
import shutil
from pathlib import Path

import cv2

from synthetic_renderer import (
    CAPTCHA_SIZE,
    random_text,
    render_text_on_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_COUNT: int = 100_000
OUTPUT_DIR: Path = Path("data/synthetic_crnn")


def generate(count: int, output_dir: Path) -> None:
    """Sinh `count` synthetic CAPTCHA + metadata.csv.

    Args:
        count: số ảnh cần sinh.
        output_dir: thư mục output.
    """
    # Xóa thư mục cũ (idempotent)
    if output_dir.exists():
        logger.info(f"Removing existing dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metadata.csv"

    rows = []
    logger.info(f"Generating {count:,} synthetic CRNN samples → {output_dir}")
    log_every = max(count // 20, 500)  # ~20 progress logs

    for i in range(count):
        text = random_text()
        img_bgr, _mask = render_text_on_image(text, CAPTCHA_SIZE)

        filename = f"captcha_{i:06d}.png"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        rows.append({"filename": filename, "text": text})

        if (i + 1) % log_every == 0 or (i + 1) == count:
            logger.info(f"  {i + 1:,}/{count:,} ({(i+1)/count*100:.1f}%)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[DONE] Saved {count:,} samples + {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CAPTCHA for CRNN training",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of samples to generate (default: {DEFAULT_COUNT:,})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if args.count < 100:
        logger.warning(
            f"Count {args.count} is very small. "
            f"Recommend ≥ 30,000 for decent CRNN training."
        )

    generate(args.count, args.output)


if __name__ == "__main__":
    main()
