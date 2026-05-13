"""
generate_data.py
================
Module tạo dataset CAPTCHA giả (synthetic) để huấn luyện mô hình TrOCR.

Render thủ công bằng Pillow để mỗi ký tự có màu ngẫu nhiên độc lập
(có thể trùng nhau), sau đó áp dụng augmentation bằng OpenCV.

Cách chạy:
    python generate_data.py
"""

import os
import random
import string
import logging
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ─── Cấu hình logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hằng số cấu hình ────────────────────────────────────────────────────────
CHARSET: str = string.ascii_uppercase + string.digits  # A-Z + 0-9
MIN_CHARS: int = 5
MAX_CHARS: int = 5
TRAIN_COUNT: int = 10_000
VAL_COUNT: int = 2_000
OUTPUT_BASE: Path = Path("data/synthetic")
CAPTCHA_WIDTH: int = 128
CAPTCHA_HEIGHT: int = 128

# Font size tương đối với chiều cao ảnh
FONT_SIZE: int = 48

# Danh sách font thử theo thứ tự ưu tiên
# Pillow sẽ dùng font đầu tiên tìm thấy, fallback về default nếu không có
_FONT_CANDIDATES = [
    "arialbd.ttf",    # Arial Bold — Windows
    "arial.ttf",      # Arial — Windows
    "DejaVuSans-Bold.ttf",  # Linux
    "LiberationSans-Bold.ttf",  # Linux
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font TrueType, fallback về default nếu không tìm thấy."""
    for font_name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    # Fallback: Pillow default bitmap font (không đẹp nhưng luôn có)
    logger.warning("Không tìm thấy TrueType font, dùng Pillow default font.")
    return ImageFont.load_default()


def random_text(min_len: int = MIN_CHARS, max_len: int = MAX_CHARS) -> str:
    """Sinh chuỗi ký tự ngẫu nhiên gồm chữ hoa và số."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(CHARSET, k=length))


def random_char_color() -> tuple[int, int, int]:
    """Sinh màu ngẫu nhiên cho 1 ký tự.

    Tránh màu quá sáng (gần trắng) để chữ vẫn nhìn thấy trên nền sáng.
    Mỗi lần gọi độc lập → các ký tự có thể trùng màu nhau.

    Returns:
        Tuple (R, G, B), mỗi kênh trong khoảng [20, 220].
    """
    return (
        random.randint(20, 220),
        random.randint(20, 220),
        random.randint(20, 220),
    )


def random_bg_color() -> tuple[int, int, int]:
    """Sinh màu nền ngẫu nhiên — sáng để chữ nổi bật."""
    return (
        random.randint(180, 255),
        random.randint(180, 255),
        random.randint(180, 255),
    )


def render_text_per_char(
    text: str,
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render từng ký tự với màu ngẫu nhiên độc lập lên ảnh.

    Mỗi ký tự được tô một màu ngẫu nhiên riêng — có thể trùng nhau.
    Vị trí mỗi ký tự có jitter nhỏ theo trục Y để trông tự nhiên hơn.

    Args:
        text: Chuỗi ký tự cần render.
        width: Chiều rộng ảnh output.
        height: Chiều cao ảnh output.
        font: Font đã load.

    Returns:
        PIL Image RGB với chữ nhiều màu trên nền gradient ngẫu nhiên.
    """
    img = Image.new("RGB", (width, height), color=random_bg_color())
    draw = ImageDraw.Draw(img)

    # Tính tổng chiều rộng để căn giữa
    char_widths = []
    for ch in text:
        bbox = font.getbbox(ch)
        char_widths.append(bbox[2] - bbox[0])

    total_text_width = sum(char_widths) + (len(text) - 1) * 4  # 4px spacing
    x_start = (width - total_text_width) // 2
    y_center = height // 2

    x = x_start
    for i, ch in enumerate(text):
        # Màu ngẫu nhiên độc lập cho mỗi ký tự (có thể trùng)
        color = random_char_color()

        # Jitter Y nhỏ để chữ không thẳng hàng hoàn toàn
        y_jitter = random.randint(-6, 6)
        bbox = font.getbbox(ch)
        char_h = bbox[3] - bbox[1]
        y = y_center - char_h // 2 + y_jitter

        # Rotation nhẹ cho từng ký tự
        angle = random.uniform(-15, 15)
        char_img = Image.new("RGBA", (char_widths[i] + 10, char_h + 10), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((5, 5), ch, font=font, fill=color + (255,))
        char_img = char_img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Paste ký tự lên ảnh chính
        img.paste(char_img, (x - 5, y - 5), char_img)

        x += char_widths[i] + 4  # spacing giữa các ký tự

    return img


def add_gradient_background(img: Image.Image) -> Image.Image:
    """Thêm gradient overlay lên nền để giống CAPTCHA thực hơn."""
    w, h = img.size
    gradient = Image.new("RGB", (w, h))
    c1 = random_bg_color()
    c2 = random_bg_color()
    for x in range(w):
        r = int(c1[0] + (c2[0] - c1[0]) * x / w)
        g = int(c1[1] + (c2[1] - c1[1]) * x / w)
        b = int(c1[2] + (c2[2] - c1[2]) * x / w)
        for y in range(h):
            gradient.putpixel((x, y), (r, g, b))

    # Blend gradient với ảnh gốc (alpha=0.3 — nhẹ thôi)
    return Image.blend(img, gradient, alpha=0.3)


def add_noise_dots(img_array: np.ndarray, num_dots: int = 80) -> np.ndarray:
    """Thêm chấm nhiễu ngẫu nhiên."""
    h, w = img_array.shape[:2]
    for _ in range(num_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        color = (
            random.randint(0, 200),
            random.randint(0, 200),
            random.randint(0, 200),
        )
        cv2.circle(img_array, (x, y), radius=random.randint(1, 2), color=color, thickness=-1)
    return img_array


def add_noise_lines(img_array: np.ndarray, num_lines: int = 5) -> np.ndarray:
    """Thêm đường kẻ nhiễu ngẫu nhiên."""
    h, w = img_array.shape[:2]
    for _ in range(num_lines):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        color = (
            random.randint(0, 180),
            random.randint(0, 180),
            random.randint(0, 180),
        )
        cv2.line(img_array, (x1, y1), (x2, y2), color, random.randint(1, 2))
    return img_array


def add_wave_distortion(img_array: np.ndarray) -> np.ndarray:
    """Biến dạng sóng sin/cos để làm méo ảnh."""
    h, w = img_array.shape[:2]

    amplitude_x = random.uniform(3, 6)
    frequency_x = random.uniform(0.05, 0.1)
    shift_x = np.sin(np.arange(h) * frequency_x) * amplitude_x

    amplitude_y = random.uniform(2, 4)
    frequency_y = random.uniform(0.05, 0.1)
    shift_y = np.cos(np.arange(w) * frequency_y) * amplitude_y

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        map_x[row, :] = np.arange(w) + shift_y
    for col in range(w):
        map_y[:, col] = np.arange(h) + shift_x

    return cv2.remap(
        img_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def generate_captcha_image(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> np.ndarray:
    """Tạo một ảnh CAPTCHA với mỗi ký tự màu ngẫu nhiên.

    Pipeline:
        1. Render từng ký tự với màu ngẫu nhiên độc lập (Pillow)
        2. Thêm gradient background overlay
        3. Thêm đường kẻ nhiễu (OpenCV)
        4. Thêm chấm nhiễu (OpenCV)
        5. Biến dạng sóng (OpenCV)

    Args:
        text: Chuỗi 5 ký tự cần render.
        font: Font đã load sẵn (tránh load lại mỗi lần).

    Returns:
        Ảnh CAPTCHA BGR numpy array (128, 128, 3).
    """
    # Bước 1: Render chữ nhiều màu
    pil_img = render_text_per_char(text, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, font)

    # Bước 2: Gradient background overlay
    pil_img = add_gradient_background(pil_img)

    # Bước 3-5: OpenCV augmentation
    img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_array = add_noise_lines(img_array, num_lines=random.randint(3, 7))
    img_array = add_noise_dots(img_array, num_dots=random.randint(50, 120))
    img_array = add_wave_distortion(img_array)

    return img_array


def generate_dataset(output_dir: Path, count: int, split_name: str) -> None:
    """Tạo dataset CAPTCHA và lưu vào thư mục chỉ định.

    Args:
        output_dir: Thư mục đầu ra.
        count: Số lượng ảnh cần tạo.
        split_name: Tên tập ('train' hoặc 'val') để log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    # Load font một lần, tái sử dụng cho toàn bộ dataset
    font = _load_font(FONT_SIZE)

    success_count = 0
    rows: list[dict[str, str]] = []

    logger.info(f"Bắt đầu tạo {count} ảnh cho tập [{split_name}]...")

    for i in range(count):
        filename = f"captcha_{i:05d}.png"
        filepath = output_dir / filename
        text = random_text()

        try:
            img_array = generate_captcha_image(text, font)
            cv2.imwrite(str(filepath), img_array)
            rows.append({"filename": filename, "text": text})
            success_count += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"  [{split_name}] Đã tạo {i + 1}/{count} ảnh...")

        except Exception as e:
            logger.error(f"  Lỗi khi tạo ảnh '{filename}': {e}")

    # Lưu metadata.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"✅ [{split_name}] Hoàn thành: {success_count}/{count} ảnh. "
        f"Metadata: {csv_path}"
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
