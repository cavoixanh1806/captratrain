# 🤖 CAPTCHA Solver — TrOCR + U-Net Pipeline

Hệ thống AI Local giải CAPTCHA 2 tầng:
- **Stage 1 — U-Net Denoiser** (~7.7M params): Nhìn từng pixel, phân loại "chữ hay nền" → xuất ảnh grayscale sạch
- **Stage 2 — TrOCR** (~334M params): Đọc chuỗi ký tự từ ảnh đã làm sạch

```
CAPTCHA (128x128, nhiễu, màu ngẫu nhiên)
       |
  ┌────▼────────┐
  │  U-Net      │  pixel nào là chữ? → soft probability map
  │  ~7.7M params│  text=đen, background=trắng
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

## Đọc hiểu output khi train

### Output U-Net (mỗi epoch):

```
Epoch  5/30 (12.3s) | Train Loss: 0.0821, IoU: 0.8234 | Val Loss: 0.0912, IoU: 0.8012, Acc: 0.9654
  -> Best model saved (IoU: 0.8234)
```

| Giá trị | Ý nghĩa | Tốt khi |
|---------|---------|---------|
| `Train Loss` | Sai số BCE khi train | Giảm dần |
| `IoU` | % vùng chữ được tách đúng (quan trọng nhất) | > 0.80 |
| `Acc` | % pixel phân loại đúng (text/background) | > 0.95 |
| `Val Loss` | Sai số trên ảnh chưa thấy | Giảm dần, gần Train Loss |

**Theo dõi: Val IoU > 0.80 là U-Net đã tốt.**

---

### Output TrOCR (mỗi 25 steps ≈ 1 epoch):

```
{'loss': 1.24, 'learning_rate': 4.2e-05, 'epoch': 4.0}
{'eval_loss': 1.31, 'eval_cer': 0.18, 'eval_exact_match': 0.72, 'epoch': 4.0}
```

| Giá trị | Ý nghĩa | Tốt khi |
|---------|---------|---------|
| `loss` | Sai số khi train | Giảm dần |
| `eval_loss` | Sai số trên ảnh chưa thấy | Giảm dần |
| `eval_cer` | % ký tự sai (Character Error Rate) | < 0.10 |
| `eval_exact_match` | % ảnh đoán đúng hoàn toàn 5/5 ký tự (quan trọng nhất) | > 0.85 |
| `learning_rate` | Tốc độ học — tự giảm dần | Không cần quan tâm |

**Theo dõi: eval_exact_match > 0.85 là model đã tốt.**

---

### Ví dụ thực tế với CER và Exact Match

```
eval_cer: 0.64  → sai 3/5 ký tự (đoán "4KAAA" thay vì "4KTN9")
eval_cer: 0.20  → sai 1/5 ký tự (đoán "4KTN7" thay vì "4KTN9")
eval_cer: 0.00  → đúng hoàn toàn

eval_exact_match: 0.72 → 72% ảnh đoán đúng cả 5 ký tự
eval_exact_match: 0.90 → 90% ảnh đoán đúng → rất tốt
```

**Lưu ý:** CAPTCHA chỉ cần 1 ký tự sai là fail hoàn toàn, nên `exact_match` là metric thực tế nhất.

---

### EarlyStopping

Training tự dừng nếu `exact_match` không cải thiện sau 8 lần eval liên tiếp (~8 epochs).
Không cần chờ hết 50 epochs — model tốt nhất được tự động lưu.

## Metrics tổng kết

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
