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
- GPU: NVIDIA RTX 3060 8GB VRAM (CUDA 12.8+)

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

Kiểm tra CUDA:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

## Chạy toàn bộ bằng 1 lệnh

```bash
venv\Scripts\activate & run_all.bat
```

Workflow tự động:
1. Extract 500 real backgrounds (inpainting xóa text)
2. Generate 24K synthetic pairs (BG thật + text mới)
3. Train U-Net denoiser (~10-15 phút)
4. Train TrOCR (~30-60 phút)
5. Evaluate trên 500 ảnh real → in kết quả

## Chạy từng bước

```bash
venv\Scripts\activate

# Bước 1: Gán nhãn 500 ảnh (nếu chưa có metadata.csv)
python label_server.py

# Bước 2: Extract backgrounds từ 500 ảnh real
python extract_real_backgrounds.py

# Bước 3: Generate U-Net training data
python generate_unet_data.py

# Bước 4: Train U-Net
python train_unet.py

# Bước 5: Train TrOCR
python train.py --use-real-data --augment

# Bước 6: Evaluate
python evaluate.py
```

## Inference

```python
from inference import CaptchaSolver

solver = CaptchaSolver(model_dir="./captcha_trocr_model")
result = solver.solve_captcha("data/map_00001.png")
print(result)  # "4KTN9"

# Batch
results = solver.solve_batch(["img1.png", "img2.png"])
```

```bash
python inference.py data/map_00001.png
```

## Đọc output khi train

### U-Net (mỗi epoch):
```
Epoch 5/30 | Train Loss: 0.082, IoU: 0.823 | Val Loss: 0.091, IoU: 0.801
```
- **Val IoU > 0.85** = U-Net tốt

### TrOCR (mỗi 25 steps):
```
{'eval_cer': 0.08, 'eval_exact_match': 0.87, 'epoch': 12.0}
```
- **eval_exact_match > 0.90** = đạt mục tiêu

### Evaluate output:
```
Exact match acc:  91.20%   ← mục tiêu ≥ 90%
CER:               4.30%   ← mục tiêu < 10%
```

## Cập nhật code mới

```bash
git pull origin master
```

## License

MIT
