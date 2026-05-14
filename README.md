# 🤖 CAPTCHA Solver — TrOCR + U-Net Pipeline

Hệ thống AI Local giải CAPTCHA 2 tầng với mục tiêu **exact_match ≥ 90%**:
- **Stage 1 — U-Net** (~7.7M params): Tách chữ khỏi nền ở mức pixel
- **Stage 2 — TrOCR** (~334M params): Đọc chuỗi ký tự

```
CAPTCHA (128x128)
       |
  ┌────▼────────┐
  │  U-Net      │  → soft probability map (text=đen, bg=trắng)
  └────┬────────┘
       |
  ┌────▼────────┐
  │  TrOCR      │  → "4KTN9"
  └─────────────┘
```

## Yêu cầu hệ thống

- Python 3.10+, RAM 16GB
- GPU: NVIDIA RTX 3060+ 8GB VRAM (CUDA 12.8+)

## Cài đặt

```bash
git clone https://github.com/cavoixanh1806/captratrain.git
cd captratrain
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install albumentations==1.4.3
```

## Chạy toàn bộ bằng 1 lệnh

```bash
venv\Scripts\activate
run_all.bat
```

Script tự động:
- Detect số ảnh `map_*.png` trong `data/` (dynamic)
- Import ảnh mới từ `dataset/` nếu có (idempotent)
- Xóa model cũ + train lại từ đầu
- Dừng ngay khi có lỗi
- Log toàn bộ ra `train_log.txt`

## Workflow

```
754 real images (data/)
       |
       ├── (auto import từ dataset/ nếu có)
       |
       ▼
[1] Generate U-Net data (12K synthetic noisy+mask pairs)
       ▼
[2] Generate TrOCR synthetic data (5K labeled samples)
       ▼
[3] Train U-Net (DiceBCE loss, 30 epochs, IoU > 0.85)
       ▼
[4] Train TrOCR (combine 5K synthetic + 600 real, 100 epochs)
       ▼
[5] Evaluate trên 754 real images
```

## Thêm data mới

Đặt ảnh mới vào `dataset/` với format: `map_<LABEL>.png`

Ví dụ: `map_4KTN9.png`, `map_WTVRY.png`

Khi chạy `run_all.bat`, script tự import vào `data/` + cập nhật `metadata.csv`. Ảnh đã có sẽ skip (check label trùng).

## Chạy từng bước

```bash
venv\Scripts\activate

# Gán nhãn (nếu chưa có metadata.csv)
python label_server.py

# Import data mới từ dataset/
python import_new_data.py

# Generate U-Net data (synthetic noisy+mask pairs)
python generate_unet_data.py

# Generate TrOCR synthetic data (có label)
python generate_trocr_synthetic.py

# Train U-Net
python train_unet.py

# Train TrOCR (combine synthetic + real)
python train.py --use-real-data --combine --augment

# Evaluate
python eval_model.py
```

## Inference

```python
from inference import CaptchaSolver

solver = CaptchaSolver(model_dir="./captcha_trocr_model")
result = solver.solve_captcha("path/to/captcha.png")
print(result)  # "4KTN9"

# Batch
results = solver.solve_batch(["img1.png", "img2.png"])
```

```bash
python inference.py data/map_00001.png
```

## Charset

CAPTCHA chỉ dùng **24 ký tự** (không phải 36):
```
Chữ: A C D E F H J K L M N P Q R T U V W X Y  (20)
Số:  3 4 7 9                                    (4)
```

Loại: O/0, I/1, S/5, B/8, G/6, Z/2 (các cặp dễ nhầm).

## Synthetic BG (calibrated từ 754 ảnh real)

Background được sinh giống real CAPTCHA:
- BGR avg (160, 157, 156) — gần pure gray
- Saturation rất thấp (avg 10)
- 70% pure gray, 24% blue-tinted, 6% other
- 31% flat, 33% mild gradient, **36% complex texture**

Không dùng inpainting (vì không xóa sạch chữ → ghost chữ).

## Đọc output khi train

### U-Net:
```
Epoch 5/30 | Train Loss: 0.082, IoU: 0.823 | Val Loss: 0.091, IoU: 0.801
```
- **Val IoU > 0.85** = tốt

### TrOCR:
```
{'eval_cer': 0.08, 'eval_exact_match': 0.87, 'epoch': 40.0}
```
- **eval_exact_match > 0.90** = đạt mục tiêu

### Evaluate:
```
Exact match acc:  91.20%
CER:               4.30%
```

## Cấu hình

| Tham số | U-Net | TrOCR |
|---------|-------|-------|
| Params | 7.7M | 334M |
| Loss | DiceBCE | CrossEntropy |
| LR | 1e-3 | 5e-5 |
| Epochs | 30 | 30 |
| Patience | — | 10 |
| Batch | 32 | 16 |
| Beams | — | 8 |
| FP16 | — | ✅ |

**Data train:**
- U-Net: 10K synthetic (BG synthetic + text render) + 2K val
- TrOCR: 2K synthetic + 600 real (754 × 80%) = 2,600 train samples

**Thời gian ước tính trên RTX 3060:** ~6 giờ tổng (U-Net 15 phút + TrOCR ~5h)
**Trên RTX 5090:** ~1.5 giờ

## Cấu trúc project

```
captratrain/
├── data/                       # 754 real ảnh + metadata.csv
├── dataset/                    # Nơi đặt ảnh mới chờ import
├── unet_model.py               # U-Net architecture
├── train_unet.py               # Train U-Net
├── generate_unet_data.py       # Sinh U-Net training pairs
├── generate_trocr_synthetic.py # Sinh TrOCR labeled data
├── import_new_data.py          # Import ảnh mới từ dataset/
├── preprocessing.py            # U-Net preprocessing wrapper
├── dataset.py                  # CaptchaDataset cho TrOCR
├── train.py                    # Train TrOCR
├── inference.py                # CaptchaSolver
├── eval_model.py               # Đánh giá model
├── label_server.py             # Web UI gán nhãn
├── run_all.bat                 # Chạy toàn bộ workflow
└── train_log.txt               # Log file (auto generated)
```

## Cập nhật code

```bash
git pull origin master
```

## License

MIT
