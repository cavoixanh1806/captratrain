"""Phân tích đặc điểm ảnh CAPTCHA thực."""
import cv2
import numpy as np
from pathlib import Path

data_dir = Path("data")
samples = sorted(data_dir.glob("map_*.png"))[:20]

print("=" * 60)
print("PHÂN TÍCH ẢNH CAPTCHA THỰC")
print("=" * 60)

for img_path in samples[:10]:
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    
    # Phân tích màu
    mean_color = img.mean(axis=(0, 1)).astype(int)
    std_color = img.std(axis=(0, 1)).astype(int)
    
    # Phân tích độ tương phản
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    
    # Phát hiện cạnh (edge) để đánh giá noise
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (255 * h * w) * 100
    
    print(f"{img_path.name}: {w}x{h}, mean_BGR={mean_color}, std={std_color}, "
          f"contrast={contrast:.1f}, edge%={edge_ratio:.1f}%")

print()
print("=" * 60)
print("SO SÁNH VỚI SYNTHETIC (thư viện captcha)")
print("=" * 60)
print(f"Ảnh thực: 128x128, background có màu (mean ~130-170)")
print(f"Synthetic: 200x80, background trắng (mean ~240-250)")
print()
print("KHÁC BIỆT CHÍNH:")
print("  1. Kích thước: Thực 128x128 (vuông) vs Synthetic 200x80 (ngang)")
print("  2. Background: Thực có màu/gradient vs Synthetic nền trắng")
print("  3. Contrast: Thực thấp hơn (chữ lẫn vào nền)")
print()

# Kiểm tra xem có ảnh nào có nền trắng không
white_count = 0
for img_path in samples:
    img = cv2.imread(str(img_path))
    if img.mean() > 200:
        white_count += 1

print(f"Ảnh nền sáng (>200): {white_count}/{len(samples)}")
print(f"Ảnh nền tối/màu (<200): {len(samples) - white_count}/{len(samples)}")
