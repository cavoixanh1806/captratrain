# 🤖 CAPTCHA Solver — TrOCR + U-Net Pipeline

Hệ thống AI Local giải CAPTCHA 2 tầng:
- **Stage 1 — U-Net Denoiser**: Tách chữ khỏi nền nhiễu ở mức pixel (soft probability map)
- **Stage 2 — TrOCR**: Fine-tune `microsoft/trocr-base-printed` để đọc chuỗi ký tự

```
CAPTCHA (128x128)
       |
  ┌────▼────────┐
  │  U-Net      │  → soft probability map (text=đen, bg=trắng)
  │  Denoiser   │
  └────┬────────┘
       |  clean image
  ┌────▼────────┐
  │  TrOCR      │  → "4KTN9"
  │  OCR        │
  └─────────────┘
```

## Branches

| Branch | Dành cho | GPU | Batch size | Tốc độ train |
|--------|----------|-----|------------|--------------|
| `master` | GPU — i5-12400F + RTX 3060 8GB | ✅ Bắt buộc | 16 | ~30-60 phút |
| `nogpu` | CPU — Xeon E3, i5 cũ... | ❌ Không cần | 4 | ~3-4 giờ |

## Đặc điểm CAPTCHA hỗ trợ

- Chữ và số (A-Z, 0-9), 5 ký tự
- Chữ dính vào nhau, méo mó, nghiêng ngả
- Background có màu (không phải nền trắng)
- Đường kẻ nhiễu, chấm nhiễu
- Mỗi chữ một màu khác nhau

## Yêu cầu hệ thống

### Branch `master` (GPU — Tối ưu cho RTX 3060 8GB + i5-12400F)
- Python 3.10+
- RAM: 16GB
- GPU: NVIDIA RTX 3060 8GB VRAM (CUDA 12.8+)
- Disk: ~5GB (model + data)

### Branch `nogpu` (CPU)
- Python 3.10+
- RAM: tối thiểu 8GB

## Cài đặt

### Clone repo

```bash
git clone https://github.com/cavoixanh1806/captratrain.git
cd captratrain
```

### Chọn branch phù hợp

```bash
# Nếu có GPU (RTX 3060+):
git checkout master

# Nếu chỉ có CPU:
git checkout nogpu
```

### Tạo môi trường ảo

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### Cài PyTorch (GPU — CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

### (Tùy chọn) Cài albumentations để augmentation mạnh hơn

```bash
pip install albumentations==1.4.3
```

Nếu không cài, hệ thống tự fallback về torchvision augmentation cơ bản.

### Kiểm tra CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Output mong đợi: CUDA: True | GPU: NVIDIA GeForce RTX 3060
```

## Quy trình sử dụng

> **Lưu ý:** Luôn kích hoạt môi trường ảo trước khi chạy:
> ```bash
> venv\Scripts\activate
> ```

### Bước 1: Gán nhãn data

Ảnh CAPTCHA nằm trong `data/`. Cần tạo file `data/metadata.csv`:

```csv
filename,text
map_00000.png,4KTN9
map_00001.png,7UTUP
```

Dùng web tool để gán nhãn nhanh:

```bash
python label_server.py
# Mở http://localhost:8080
```

### Bước 2 (Tùy chọn): Train U-Net Denoiser

Nếu muốn dùng preprocessing tốt nhất (`--preprocess unet`):

```bash
# Tạo 12,000 ảnh synthetic cho U-Net
python generate_unet_data.py

# Train U-Net (~10-15 phút trên RTX 3060)
python train_unet.py
```

Nếu bỏ qua bước này, dùng `--preprocess color` hoặc `--preprocess combined` thay thế.

### Bước 3: Train TrOCR

**Lệnh tối ưu cho RTX 3060 8GB + i5-12400F + 500 ảnh:**

```bash
# Với U-Net preprocessing (tốt nhất — cần Bước 2):
python train.py --use-real-data --preprocess unet --augment

# Không có U-Net (nhanh hơn, ít chính xác hơn):
python train.py --use-real-data --preprocess color --augment

# Kết hợp synthetic + real data:
python train.py --use-real-data --combine --preprocess unet --augment
```

Model lưu vào `./captcha_trocr_model/`

### Bước 4: Dự đoán CAPTCHA

```bash
python inference.py data/map_00050.png
# Output: Kết quả: AB3K7
```

```python
from inference import CaptchaSolver

# Single image
solver = CaptchaSolver(preprocess_method="unet")  # hoặc "color"
result = solver.solve_captcha("path/to/captcha.png")
print(result)  # "AB3K7"

# Batch inference (nhanh hơn khi giải nhiều ảnh)
results = solver.solve_batch(["img1.png", "img2.png", "img3.png"])
print(results)  # ["AB3K7", "4KTN9", "7UTUP"]
```

## Tham số huấn luyện (tối ưu cho phần cứng)

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `BATCH_SIZE` | 16 | 400 train / 16 = 25 steps/epoch, vừa VRAM 8GB |
| `LEARNING_RATE` | 5e-5 | Phù hợp fine-tuning TrOCR |
| `NUM_EPOCHS` | 50 | 50 × 25 = 1250 total steps |
| `warmup_steps` | 100 | ~8% total steps |
| `MAX_TARGET_LENGTH` | 8 | [BOS] + 5 chars + [EOS] = 7, dư 1 |
| `fp16` | ✅ | Mixed precision — tiết kiệm VRAM RTX 3060 |
| `dataloader_num_workers` | 4 | i5-12400F 6c/12t, dùng 4 workers |
| `EarlyStopping patience` | 8 | Dừng sau 8 lần eval không cải thiện (~8 epochs) |
| `num_beams` | 4 | Beam search — cân bằng tốc độ/chính xác |

## Preprocessing methods

| Method | Mô tả | Yêu cầu | Độ chính xác |
|--------|--------|---------|--------------|
| `unet` | U-Net soft probability map | Train U-Net trước | ⭐⭐⭐⭐⭐ |
| `combined` | Color + Adaptive threshold | Không | ⭐⭐⭐⭐ |
| `color` | HSV color segmentation | Không | ⭐⭐⭐ |
| `enhanced` | CLAHE contrast + sharpen | Không | ⭐⭐⭐ |
| `adaptive` | Adaptive threshold | Không | ⭐⭐ |

**Điểm khác biệt của `unet`**: Thay vì binarize threshold cứng (mất thông tin),
U-Net xuất ra soft probability map — chữ có độ xám tỷ lệ với xác suất là text pixel.
Kết quả: ảnh grayscale sạch với soft edges, TrOCR đọc chính xác hơn.

## Augmentation

Khi dùng `--augment`, hệ thống tự chọn augmentation mạnh nhất có thể:

| Thư viện | Augmentation |
|----------|-------------|
| `albumentations` (nếu có) | ElasticTransform, GridDistortion, GaussNoise, MotionBlur, ColorJitter |
| `torchvision` (fallback) | Rotation, ColorJitter, GaussianBlur |

Augmentation albumentations giống nhiễu CAPTCHA thực tế hơn → khuyến nghị cài.

## Metrics

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|---------|
| `exact_match` | % ảnh đoán đúng hoàn toàn 5/5 ký tự | > 0.85 |
| `cer` | Tỷ lệ ký tự sai (Character Error Rate) | < 0.10 |

**Tại sao dùng `exact_match` thay vì chỉ `cer`?**
CAPTCHA chỉ cần 1 ký tự sai là fail hoàn toàn. CER = 0.2 (sai 1/5 ký tự) vẫn là
fail 100% trong thực tế. `exact_match` phản ánh đúng hiệu quả thực tế hơn.

Model được lưu checkpoint tốt nhất dựa trên `exact_match` cao nhất.

## Giải thích output khi train

```
{'loss': 1.24, 'learning_rate': 4.2e-05, 'epoch': 8.0}
{'eval_loss': 1.31, 'eval_cer': 0.18, 'eval_exact_match': 0.72, 'epoch': 8.0}
```

| Giá trị | Ý nghĩa | Tốt khi nào? |
|---------|---------|--------------|
| `loss` | Sai số khi train | Giảm dần |
| `eval_cer` | Tỷ lệ ký tự sai trên val set | < 0.10 |
| `eval_exact_match` | % ảnh đoán đúng hoàn toàn | > 0.85 |
| `epoch` | Số lần lặp qua toàn bộ data | Chỉ để theo dõi |

## Cấu trúc project

```
captratrain/
├── unet_model.py          # U-Net architecture (2M params)
├── preprocessing.py       # 5 preprocessing methods (unet/color/enhanced/adaptive/combined)
├── dataset.py             # CaptchaDataset + augmentation pipeline
├── train.py               # Fine-tune TrOCR (Seq2SeqTrainer)
├── inference.py           # CaptchaSolver class (single + batch)
├── generate_unet_data.py  # Tạo 12K synthetic pairs cho U-Net
├── train_unet.py          # Train U-Net denoiser
├── generate_data.py       # Tạo synthetic CAPTCHA cho TrOCR
├── label_server.py        # Web UI gán nhãn
├── data/
│   ├── map_*.png          # 500 ảnh CAPTCHA thực
│   └── metadata.csv       # Nhãn thủ công (filename, text)
├── captcha_trocr_model/   # Model TrOCR đã train (sau khi train xong)
└── captcha_unet_model.pth # Model U-Net đã train (sau khi train xong)
```

## Cập nhật code mới

```bash
git pull origin master
```

## License

MIT
