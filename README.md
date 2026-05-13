# 🤖 CAPTCHA Solver — TrOCR + U-Net Pipeline

Hệ thống AI Local giải CAPTCHA 2 tầng:
- **Stage 1 — U-Net Denoiser** (~2M params): Nhìn từng pixel, phân loại "chữ hay nền" → xuất ảnh grayscale sạch
- **Stage 2 — TrOCR** (~334M params): Đọc chuỗi ký tự từ ảnh đã làm sạch

```
CAPTCHA (128x128, nhiễu, màu ngẫu nhiên)
       |
  ┌────▼────────┐
  │  U-Net      │  pixel nào là chữ? → soft probability map
  │  ~2M params │  text=đen, background=trắng
  └────┬────────┘
       |  ảnh grayscale sạch
  ┌────▼────────┐
  │  TrOCR      │  → "4KTN9"
  │  ~334M params│
  └─────────────┘
```

**Tại sao dùng U-Net?**
U-Net học từ 12K ảnh synthetic nên hiểu hình dạng chữ, không phụ thuộc màu sắc.
Dù ký tự màu gì, trùng nhau hay không, U-Net vẫn tách đúng vì nó học pattern hình dạng.

## Yêu cầu hệ thống

- Python 3.10+
- RAM: 16GB
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
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

## Quy trình train

> Luôn kích hoạt môi trường ảo trước:
> ```bash
> venv\Scripts\activate
> ```

### Bước 1 — Gán nhãn 500 ảnh thực

```bash
python label_server.py
# Mở http://localhost:8080
```

### Bước 2 — Train U-Net Denoiser

```bash
python generate_unet_data.py
python train_unet.py
```

### Bước 3 — Train TrOCR

```bash
python train.py --use-real-data --augment
```

### Hoặc chạy tất cả Bước 2 + 3 bằng 1 lệnh

```bash
venv\Scripts\activate & python generate_unet_data.py & python train_unet.py & python train.py --use-real-data --augment
```

## Inference

```python
from inference import CaptchaSolver

solver = CaptchaSolver(model_dir="./captcha_trocr_model")
result = solver.solve_captcha("data/map_00001.png")
print(result)  # "4KTN9"

# Batch — nhanh hơn khi giải nhiều ảnh cùng lúc
results = solver.solve_batch(["img1.png", "img2.png"])
print(results)  # ["4KTN9", "AB3K7"]
```

```bash
python inference.py data/map_00001.png
```

## Metrics

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|---------|
| `exact_match` | % ảnh đoán đúng hoàn toàn 5/5 ký tự | > 0.85 |
| `cer` | Tỷ lệ ký tự sai | < 0.10 |

## Cập nhật code mới

```bash
git pull origin master
```

## License

MIT
