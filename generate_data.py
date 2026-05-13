"""
generate_data.py
================
Module tạo dataset CAPTCHA giả (synthetic) để huấn luyện mô hình TrOCR.

Sử dụng thư viện `captcha` để sinh ảnh, sau đó áp dụng thêm các kỹ thuật
augmentation bằng OpenCV và Pillow để làm ảnh méo mó, thêm nhiễu,
giống với CAPTCHA thực tế hơn.

Cách chạy:
    python generate_data.py
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
from PIL import Image
from captcha.image import ImageCaptcha

# ─── Cấu hình logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hằng số cấu hình ────────────────────────────────────────────────────────
CHARSET: str = string.ascii_uppercase + string.digits  # A-Z + 0-9
MIN_CHARS: int = 4
MAX_CHARS: int = 6
TRAIN_COUNT: int = 10_000
VAL_COUNT: int = 2_000
OUTPUT_BASE: Path = Path("data/synthetic")
CAPTCHA_WIDTH: int = 200
CAPTCHA_HEIGHT: int = 80


def random_text(min_len: int = MIN_CHARS, max_len: int = MAX_CHARS) -> str:
    """Sinh chuỗi ký tự ngẫu nhiên gồm chữ hoa và số.

    Args:
        min_len: Độ dài tối thiểu của chuỗi.
        max_len: Độ dài tối đa của chuỗi.

    Returns:
        Chuỗi ký tự ngẫu nhiên.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choices(CHARSET, k=length))


def add_noise_dots(img_array: np.ndarray, num_dots: int = 80) -> np.ndarray:
    """Thêm chấm nhiễu (noise dots) ngẫu nhiên lên ảnh.

    Args:
        img_array: Ảnh dạng numpy array (H, W, C).
        num_dots: Số lượng chấm nhiễu cần thêm.

    Returns:
        Ảnh đã thêm chấm nhiễu.
    """
    h, w = img_array.shape[:2]
    for _ in range(num_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        color = (
            random.randint(0, 150),
            random.randint(0, 150),
            random.randint(0, 150),
        )
        cv2.circle(img_array, (x, y), radius=1, color=color, thickness=-1)
    return img_array


def add_noise_lines(img_array: np.ndarray, num_lines: int = 5) -> np.ndarray:
    """Thêm đường kẻ nhiễu (noise lines) ngẫu nhiên lên ảnh.

    Args:
        img_array: Ảnh dạng numpy array (H, W, C).
        num_lines: Số lượng đường kẻ nhiễu cần thêm.

    Returns:
        Ảnh đã thêm đường kẻ nhiễu.
    """
    h, w = img_array.shape[:2]
    for _ in range(num_lines):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        color = (
            random.randint(0, 180),
            random.randint(0, 180),
            random.randint(0, 180),
        )
        thickness = random.randint(1, 2)
        cv2.line(img_array, (x1, y1), (x2, y2), color, thickness)
    return img_array


def add_wave_distortion(img_array: np.ndarray) -> np.ndarray:
    """Áp dụng biến dạng sóng (wave distortion) để làm méo ảnh.

    Kỹ thuật này dùng sin/cos để dịch chuyển pixel theo trục X và Y,
    tạo hiệu ứng méo mó giống CAPTCHA thực tế.

    Args:
        img_array: Ảnh dạng numpy array (H, W, C).

    Returns:
        Ảnh đã bị biến dạng sóng.
    """
    h, w = img_array.shape[:2]

    # Tạo map dịch chuyển theo trục X (sóng ngang)
    amplitude_x = random.uniform(3, 6)
    frequency_x = random.uniform(0.05, 0.1)
    shift_x = np.sin(np.arange(h) * frequency_x) * amplitude_x

    # Tạo map dịch chuyển theo trục Y (sóng dọc)
    amplitude_y = random.uniform(2, 4)
    frequency_y = random.uniform(0.05, 0.1)
    shift_y = np.cos(np.arange(w) * frequency_y) * amplitude_y

    # Xây dựng map tọa độ mới
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        map_x[row, :] = np.arange(w) + shift_y
    for col in range(w):
        map_y[:, col] = np.arange(h) + shift_x

    # Áp dụng remap để biến dạng ảnh
    distorted = cv2.remap(
        img_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return distorted


def generate_captcha_image(text: str) -> np.ndarray:
    """Tạo một ảnh CAPTCHA từ chuỗi text và áp dụng augmentation.

    Pipeline:
        1. Sinh ảnh CAPTCHA cơ bản bằng thư viện `captcha`
        2. Thêm đường kẻ nhiễu
        3. Thêm chấm nhiễu
        4. Áp dụng biến dạng sóng

    Args:
        text: Chuỗi ký tự cần render lên ảnh CAPTCHA.

    Returns:
        Ảnh CAPTCHA đã augment dạng numpy array (H, W, C) BGR.
    """
    # Bước 1: Sinh ảnh CAPTCHA cơ bản
    generator = ImageCaptcha(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT)
    img_bytes = generator.generate(text)
    pil_img = Image.open(BytesIO(img_bytes.read())).convert("RGB")

    # Chuyển sang numpy array (BGR cho OpenCV)
    img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Bước 2: Thêm đường kẻ nhiễu
    img_array = add_noise_lines(img_array, num_lines=random.randint(3, 7))

    # Bước 3: Thêm chấm nhiễu
    img_array = add_noise_dots(img_array, num_dots=random.randint(50, 120))

    # Bước 4: Biến dạng sóng
    img_array = add_wave_distortion(img_array)

    return img_array


def generate_dataset(output_dir: Path, count: int, split_name: str) -> None:
    """Tạo dataset CAPTCHA và lưu vào thư mục chỉ định.

    Args:
        output_dir: Thư mục đầu ra để lưu ảnh và metadata.csv.
        count: Số lượng ảnh cần tạo.
        split_name: Tên tập dữ liệu ('train' hoặc 'val') để log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    success_count = 0
    rows: list[dict[str, str]] = []

    logger.info(f"Bắt đầu tạo {count} ảnh cho tập [{split_name}]...")

    for i in range(count):
        filename = f"captcha_{i:05d}.png"
        filepath = output_dir / filename
        text = random_text()

        try:
            img_array = generate_captcha_image(text)
            cv2.imwrite(str(filepath), img_array)
            rows.append({"filename": filename, "text": text})
            success_count += 1

            # In tiến trình mỗi 1000 ảnh
            if (i + 1) % 1000 == 0:
                logger.info(f"  [{split_name}] Đã tạo {i + 1}/{count} ảnh...")

        except Exception as e:
            # Ghi log lỗi và tiếp tục (không dừng toàn bộ quá trình)
            logger.error(f"  Lỗi khi tạo ảnh '{filename}': {e}")

    # Lưu metadata.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"✅ [{split_name}] Hoàn thành: {success_count}/{count} ảnh. "
        f"Metadata lưu tại: {csv_path}"
    )


def main() -> None:
    """Hàm chính: tạo cả tập train và val."""
    train_dir = OUTPUT_BASE / "train"
    val_dir = OUTPUT_BASE / "val"

    generate_dataset(train_dir, TRAIN_COUNT, "train")
    generate_dataset(val_dir, VAL_COUNT, "val")

    logger.info("🎉 Tạo dataset hoàn tất!")
    logger.info(f"   Train: {train_dir}")
    logger.info(f"   Val:   {val_dir}")


if __name__ == "__main__":
    main()
