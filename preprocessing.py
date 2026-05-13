"""
preprocessing.py
=================
Module tien xu ly anh CAPTCHA truoc khi dua vao TrOCR.

Phuong phap:
- adaptive: Adaptive threshold
- color: HSV color segmentation
- combined: color + adaptive
- enhanced: CLAHE contrast + sharpen
- unet: U-Net denoiser (pixel-level probability map → soft grayscale)

Pipeline: CAPTCHA -> Preprocessing -> TrOCR -> Text

FIX: unet method nay dung truc tiep probability map lam grayscale thay vi
     binary threshold + lay pixel goc (mat thong tin gradient).
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Optional

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


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    grid_size: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Tăng contrast cục bộ bằng CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE chia ảnh thành các ô nhỏ (grid_size), equalize histogram trên từng ô,
    rồi nội suy kết quả → tăng contrast cục bộ mà không bị quá sáng/tối.

    Args:
        image: Ảnh BGR đầu vào.
        clip_limit: Ngưỡng giới hạn contrast (cao hơn = contrast mạnh hơn).
        grid_size: Kích thước lưới chia ô.

    Returns:
        Ảnh đã tăng contrast.
    """
    # Chuyển sang LAB color space — tách riêng kênh L (lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Áp CLAHE lên kênh L (lightness) để tăng contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Ghép lại và chuyển về BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def remove_background(
    image: np.ndarray,
    block_size: int = 15,
    c_value: int = 8,
) -> np.ndarray:
    """Tách chữ khỏi nền bằng Adaptive Thresholding.

    Adaptive threshold xử lý từng vùng nhỏ → hiệu quả với nền gradient/màu
    (không giống global threshold chỉ dùng 1 ngưỡng cho cả ảnh).

    Args:
        image: Ảnh BGR đầu vào.
        block_size: Kích thước vùng lân cận để tính threshold (phải lẻ).
        c_value: Hằng số trừ đi từ mean (cao hơn = lọc mạnh hơn).

    Returns:
        Ảnh nhị phân (đen trắng), chữ = đen, nền = trắng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: tính ngưỡng cho từng vùng nhỏ block_size x block_size
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )
    return binary


def remove_noise_lines(
    binary: np.ndarray,
    min_line_length: int = 20,
) -> np.ndarray:
    """Loại bỏ đường kẻ nhiễu bằng morphological operations.

    Sử dụng 2 kỹ thuật:
      1. Opening (erosion → dilation): loại bỏ chấm nhỏ
      2. Connected component analysis: loại bỏ thành phần quá nhỏ

    Args:
        binary: Ảnh nhị phân đầu vào.
        min_line_length: Độ dài tối thiểu để giữ lại thành phần.

    Returns:
        Ảnh đã khử nhiễu.
    """
    # Đảo ngược: chữ = trắng, nền = đen (để morphology xử lý chữ)
    inverted = cv2.bitwise_not(binary)

    # Kernel nhỏ cho opening — loại bỏ chấm nhiễu nhỏ
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Dilation nhẹ để nối các phần chữ bị đứt
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)

    # Đảo lại: chữ = đen, nền = trắng
    result = cv2.bitwise_not(cleaned)
    return result


def extract_text_by_color(
    image: np.ndarray,
    saturation_threshold: int = 40,
    value_range: tuple[int, int] = (30, 200),
) -> np.ndarray:
    """Tách chữ dựa trên màu sắc (HSV color segmentation).

    CAPTCHA này có đặc điểm: chữ có màu sặc sỡ (đỏ, xanh, vàng...),
    nền thì nhạt/gradient. Dùng HSV để tách vùng có saturation cao (chữ).

    Args:
        image: Ảnh BGR đầu vào.
        saturation_threshold: Ngưỡng saturation tối thiểu để coi là "chữ".
        value_range: Khoảng value (brightness) cho phép.

    Returns:
        Mask nhị phân: chữ = trắng, nền = đen.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Chữ thường có saturation cao hơn nền
    mask_saturation = s > saturation_threshold

    # Loại bỏ vùng quá tối hoặc quá sáng
    mask_value = (v > value_range[0]) & (v < value_range[1])

    # Kết hợp 2 mask
    mask = (mask_saturation & mask_value).astype(np.uint8) * 255

    # Làm mịn mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def preprocess_captcha(
    image: np.ndarray,
    method: str = "combined",
) -> Image.Image:
    """Pipeline tiền xử lý chính cho CAPTCHA.

    Args:
        image: BGR numpy array.
        method: Preprocessing method:
            - "adaptive": adaptive threshold only
            - "color": HSV color segmentation
            - "combined": color + adaptive (tốt nhất không có U-Net)
            - "enhanced": CLAHE contrast + sharpen
            - "unet": U-Net denoiser — dùng soft probability map trực tiếp
                      (tốt nhất, giữ nguyên thông tin gradient thay vì binarize)

    Returns:
        PIL RGB image.
    """
    if method == "unet":
        # ── U-Net denoiser: dùng soft probability map làm grayscale ──────────
        # FIX: Không binarize threshold=0.5 (mất thông tin).
        # Thay vào đó: prob_map → invert → grayscale RGB → TrOCR
        # prob=1.0 (text pixel) → intensity=0 (đen)
        # prob=0.0 (background) → intensity=255 (trắng)
        # Kết quả: ảnh grayscale sạch, chữ đen trên nền trắng, giữ soft edges.
        model, device = _get_unet_model()

        # Convert BGR -> RGB, normalize to [0, 1], add batch dim
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 128, 128)

        # Predict pixel probabilities
        with torch.no_grad():
            prob_map = model(tensor)  # (1, 1, 128, 128), values 0-1

        # Dùng soft probability map trực tiếp — không binarize
        # Invert: text (prob≈1) → dark (0), background (prob≈0) → white (255)
        prob_np = prob_map.squeeze().cpu().numpy()  # (128, 128), float 0-1
        clean_gray = ((1.0 - prob_np) * 255.0).astype(np.uint8)

        # Stack thành RGB (3 kênh) để TrOCR processor nhận đúng format
        clean_rgb = np.stack([clean_gray, clean_gray, clean_gray], axis=-1)
        return Image.fromarray(clean_rgb)

    elif method == "enhanced":
        # Chỉ tăng contrast, giữ màu gốc — phù hợp nhất cho TrOCR
        enhanced = enhance_contrast_clahe(image, clip_limit=4.0, grid_size=(4, 4))

        # Sharpen — làm rõ cạnh chữ
        kernel_sharpen = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        result = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

    elif method == "adaptive":
        enhanced = enhance_contrast_clahe(image)
        binary = remove_background(enhanced, block_size=15, c_value=8)
        cleaned = remove_noise_lines(binary)
        # Chuyển grayscale → RGB (3 kênh) cho TrOCR
        rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    elif method == "color":
        enhanced = enhance_contrast_clahe(image, clip_limit=4.0)
        mask = extract_text_by_color(enhanced, saturation_threshold=35)
        # Tạo ảnh nền trắng + chữ đen
        result = np.full_like(image, 255)
        result[mask > 0] = image[mask > 0]
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    elif method == "combined":
        # === Phương pháp kết hợp (tốt nhất khi không có U-Net) ===
        enhanced = enhance_contrast_clahe(image, clip_limit=4.0)

        # Mask 1: Color-based (tách chữ có màu sặc sỡ)
        color_mask = extract_text_by_color(enhanced, saturation_threshold=30)

        # Mask 2: Adaptive threshold (tách chữ theo intensity)
        binary = remove_background(enhanced, block_size=17, c_value=6)
        binary_inv = cv2.bitwise_not(binary)  # Đảo: chữ = trắng

        # Kết hợp 2 mask bằng OR — giữ lại pixel nào xuất hiện ở ít nhất 1 mask
        combined_mask = cv2.bitwise_or(color_mask, binary_inv)

        # Khử nhiễu nhỏ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Tạo ảnh kết quả: nền trắng + chữ giữ màu gốc
        result = np.full_like(enhanced, 255)  # Nền trắng
        result[combined_mask > 0] = enhanced[combined_mask > 0]

        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    else:
        raise ValueError(f"Method không hợp lệ: {method}. "
                         f"Chọn: adaptive, color, combined, enhanced, unet")


def preprocess_from_path(
    image_path: str | Path,
    method: str = "combined",
) -> Image.Image:
    """Tiện ích: đọc ảnh từ path và preprocess.

    Args:
        image_path: Đường dẫn ảnh CAPTCHA.
        method: Phương pháp preprocessing.

    Returns:
        Ảnh PIL đã tiền xử lý.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return preprocess_captcha(img, method=method)


def compare_methods(image_path: str | Path, output_dir: str | Path = "data/preprocessed") -> None:
    """So sánh tất cả phương pháp preprocessing trên 1 ảnh.

    Lưu kết quả vào thư mục output_dir để so sánh trực quan.

    Args:
        image_path: Đường dẫn ảnh gốc.
        output_dir: Thư mục lưu kết quả.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    stem = image_path.stem

    methods = ["enhanced", "adaptive", "color", "combined"]
    # Only include unet if model exists
    if Path(UNET_MODEL_PATH).exists():
        methods.append("unet")
    for method in methods:
        result = preprocess_captcha(img, method=method)
        out_path = output_dir / f"{stem}_{method}.png"
        result.save(str(out_path))
        print(f"  [OK] {method:10s} -> {out_path}")


if __name__ == "__main__":
    """Demo: so sánh các phương pháp trên 5 ảnh đầu tiên."""
    import sys

    data_dir = Path("data")
    output_dir = Path("data/preprocessed")

    # Lấy 5 ảnh đầu để demo
    samples = sorted(data_dir.glob("map_*.png"))[:5]

    if not samples:
        print("Không tìm thấy ảnh trong data/")
        sys.exit(1)

    print("=" * 60)
    print("COMPARE PREPROCESSING METHODS")
    print("=" * 60)

    for img_path in samples:
        print(f"\n[IMG] {img_path.name}:")
        compare_methods(img_path, output_dir)

    print(f"\n[DONE] Results saved to: {output_dir}/")
    print("   Open that folder to compare visually.")
