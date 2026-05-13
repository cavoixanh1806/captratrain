"""
generate_unet_data.py
=====================
Generate paired training data for U-Net CAPTCHA denoiser.

For each sample, creates:
    - Noisy image: CAPTCHA with background, noise lines, distortion (128x128)
    - Clean mask: Binary mask where text pixels = 255, background = 0 (128x128)

The noisy image simulates real CAPTCHA characteristics:
    - Colored/gradient background (similar to real CAPTCHAs)
    - Random noise lines and dots
    - Wave distortion
    - Rotated, multi-colored characters

Usage:
    python generate_unet_data.py
"""

import os
import random
import string
import logging
import csv
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHARSET: str = string.ascii_uppercase + string.digits
CAPTCHA_SIZE: int = 128       # Output size (128x128 to match real CAPTCHAs)
TRAIN_COUNT: int = 10_000
VAL_COUNT: int = 2_000
OUTPUT_BASE: Path = Path("data/unet_pairs")


def random_text(length: int = 5) -> str:
    """Generate random CAPTCHA text."""
    return "".join(random.choices(CHARSET, k=length))


def generate_gradient_background(size: int = 128) -> np.ndarray:
    """Generate a colored gradient background similar to real CAPTCHAs.

    Creates gradient with random direction and colors to simulate
    the map-like backgrounds in real CAPTCHA images.

    Args:
        size: Image size (square).

    Returns:
        BGR numpy array (size, size, 3).
    """
    # Random base colors for gradient
    color1 = np.array([random.randint(100, 200) for _ in range(3)], dtype=np.float32)
    color2 = np.array([random.randint(100, 200) for _ in range(3)], dtype=np.float32)

    # Random gradient direction
    direction = random.choice(["horizontal", "vertical", "diagonal"])

    img = np.zeros((size, size, 3), dtype=np.float32)

    for i in range(size):
        if direction == "horizontal":
            ratio = i / size
        elif direction == "vertical":
            ratio = i / size
        else:
            ratio = i / size

        color = color1 * (1 - ratio) + color2 * ratio

        if direction == "horizontal":
            img[:, i] = color
        elif direction == "vertical":
            img[i, :] = color
        else:
            img[i, :] = color * (1 - i / size) + color2 * (i / size)

    # Add slight noise to background
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    return img.astype(np.uint8)


def render_text_on_image(
    text: str,
    size: int = 128,
    render_on_bg: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Render CAPTCHA text and return both noisy image and clean mask.

    Args:
        text: Text to render.
        size: Image size.
        render_on_bg: If True, render on gradient background.

    Returns:
        Tuple of (noisy_image, clean_mask), both (size, size, 3) BGR.
    """
    # ── Create clean text mask on white background ────────────────────────────
    mask_pil = Image.new("L", (size, size), 0)
    text_pil = Image.new("RGB", (size, size), (255, 255, 255))

    mask_draw = ImageDraw.Draw(mask_pil)
    text_draw = ImageDraw.Draw(text_pil)

    # Character-by-character rendering with random transforms
    char_width = size // (len(text) + 1)
    start_x = random.randint(5, 15)

    for i, char in enumerate(text):
        font_size = random.randint(22, 36)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Random position with some variation
        x = start_x + i * char_width + random.randint(-3, 3)
        y = random.randint(size // 4 - 10, size // 4 + 15)

        # Random color for text (saturated colors)
        text_color = (
            random.randint(20, 200),
            random.randint(20, 200),
            random.randint(20, 200),
        )

        # Render on mask (white text on black bg)
        mask_draw.text((x, y), char, fill=255, font=font)

        # Render colored text
        text_draw.text((x, y), char, fill=text_color, font=font)

    # Rotate individual characters by rotating the entire image slightly
    angle = random.uniform(-5, 5)
    mask_pil = mask_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    text_pil = text_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    # ── Create noisy version ──────────────────────────────────────────────────
    if render_on_bg:
        bg = generate_gradient_background(size)
    else:
        bg = np.full((size, size, 3), 200, dtype=np.uint8)

    # Convert mask to numpy for blending
    mask_np = np.array(mask_pil)
    text_np = cv2.cvtColor(np.array(text_pil), cv2.COLOR_RGB2BGR)

    # Blend: where mask > threshold, use text pixels; otherwise use background
    mask_3ch = np.stack([mask_np] * 3, axis=-1) / 255.0
    noisy = (text_np * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)

    # Add noise lines
    num_lines = random.randint(3, 7)
    for _ in range(num_lines):
        x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
        x2, y2 = random.randint(0, size - 1), random.randint(0, size - 1)
        color = tuple(random.randint(50, 200) for _ in range(3))
        thickness = random.randint(1, 2)
        cv2.line(noisy, (x1, y1), (x2, y2), color, thickness)

    # Add noise dots
    num_dots = random.randint(30, 80)
    for _ in range(num_dots):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        color = tuple(random.randint(0, 200) for _ in range(3))
        cv2.circle(noisy, (x, y), 1, color, -1)

    # Wave distortion
    h, w = noisy.shape[:2]
    amp_x = random.uniform(2, 4)
    freq_x = random.uniform(0.04, 0.08)
    amp_y = random.uniform(1, 3)
    freq_y = random.uniform(0.04, 0.08)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    shift_x = np.sin(np.arange(h) * freq_x) * amp_x
    shift_y = np.cos(np.arange(w) * freq_y) * amp_y

    for row in range(h):
        map_x[row, :] = np.arange(w) + shift_y
    for col in range(w):
        map_y[:, col] = np.arange(h) + shift_x

    noisy = cv2.remap(noisy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Apply same distortion to mask for consistency
    mask_3ch_uint8 = (mask_np).astype(np.float32)
    mask_distorted = cv2.remap(mask_3ch_uint8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_final = (mask_distorted > 50).astype(np.uint8) * 255

    return noisy, mask_final


def generate_dataset(output_dir: Path, count: int, split_name: str) -> None:
    """Generate paired dataset for U-Net training.

    Args:
        output_dir: Output directory.
        count: Number of pairs to generate.
        split_name: "train" or "val".
    """
    noisy_dir = output_dir / "noisy"
    mask_dir = output_dir / "mask"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {count} pairs for [{split_name}]...")

    for i in range(count):
        text = random_text()
        noisy, mask = render_text_on_image(text, CAPTCHA_SIZE)

        filename = f"unet_{i:05d}.png"
        cv2.imwrite(str(noisy_dir / filename), noisy)
        cv2.imwrite(str(mask_dir / filename), mask)

        if (i + 1) % 2000 == 0:
            logger.info(f"  [{split_name}] {i + 1}/{count} pairs done")

    logger.info(f"  [{split_name}] Done: {count} pairs saved")


def main() -> None:
    """Generate train and val datasets."""
    train_dir = OUTPUT_BASE / "train"
    val_dir = OUTPUT_BASE / "val"

    generate_dataset(train_dir, TRAIN_COUNT, "train")
    generate_dataset(val_dir, VAL_COUNT, "val")

    logger.info(f"[DONE] U-Net training data saved to: {OUTPUT_BASE}")
    logger.info(f"  Train: {TRAIN_COUNT} pairs")
    logger.info(f"  Val:   {VAL_COUNT} pairs")


if __name__ == "__main__":
    main()
