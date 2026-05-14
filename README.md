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

Mở CMD, chạy:
```bash
venv\Scripts\activate
run_all.bat
```

Script tự động:
- Xóa model cũ + train lại từ đầu
- Detect số ảnh `map_*.png` trong `data/` (dynamic)
- Import ảnh mới từ `dataset/` nếu có
- Dừng ngay khi có lỗi + hiện thông báo
- Log toàn bộ output ra `train_log.txt`

## Thêm data mới

Đặt ảnh mới vào thư mục `dataset/` với format tên: `map_<LABEL>.png`

Ví dụ: `map_4KTN9.png`, `map_WTVRY.png`

Khi chạy `run_all.bat`, script tự import vào `data/` + cập nhật `metadata.csv`.

Hoặc import thủ công:
```bash
python import_new_data.py
```

## Chạy từng bước

```bash
venv\Scripts\activate

# Gán nhãn (nếu chưa có metadata.csv)
python label_server.py

# Extract backgrounds
python extract_real_backgrounds.py

# Generate U-Net data
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
| Epochs | 30 | 100 |
| Patience | — | 20 |
| Batch | 32 | 16 |
| Beams | — | 8 |
| FP16 | — | ✅ |

## Cập nhật code

```bash
git pull origin master
```

## License

MIT
