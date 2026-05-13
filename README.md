# 🤖 CAPTCHA Solver — Fine-tuning TrOCR

Hệ thống AI Local giải CAPTCHA bằng phương pháp Fine-tuning mô hình TrOCR (`microsoft/trocr-base-printed`) của Microsoft.

## Branches

| Branch | Dành cho | GPU | Batch size | Tốc độ train |
|---|---|---|---|---|
| `master` | GPU — i5-12400F + RTX 3060+ | ✅ Bắt buộc | 32 | ~10-15 phút |
| `nogpu` | CPU — Xeon E3, i5 cũ... | ❌ Không cần | 4 | ~3-4 giờ |

## Đặc điểm CAPTCHA hỗ trợ

- Chữ và số (A-Z, 0-9), 5 ký tự
- Chữ dính vào nhau, méo mó, nghiêng ngả
- Background có màu (không phải nền trắng)
- Đường kẻ nhiễu, chấm nhiễu
- Mỗi chữ một màu khác nhau

## Yêu cầu hệ thống

### Branch `master` (GPU)
- Python 3.10+
- RAM: 16GB (dùng tối đa 10GB)
- GPU: NVIDIA RTX 3060+ với CUDA 11.8+
- VRAM: tối thiểu 8GB

### Branch `nogpu` (CPU)
- Python 3.10+
- RAM: tối thiểu 8GB

## Cài đặt

```bash
# Clone repo
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

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Cài PyTorch

**Branch `master` (GPU — CUDA 12.8, tương thích CUDA 13.x):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Branch `nogpu` (CPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

### Kiểm tra CUDA (branch master)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Output mong đợi: CUDA: True | GPU: NVIDIA GeForce RTX 3060
```

## Sử dụng

> **Lưu ý:** Luôn kích hoạt môi trường ảo trước khi chạy bất kỳ lệnh nào:
> ```bash
> # Windows
> venv\Scripts\activate
> # Linux/Mac
> source venv/bin/activate
> ```

### 1. Gán nhãn data

Ảnh CAPTCHA nằm trong `data/`. Cần tạo file `data/metadata.csv` với format:

```csv
filename,text
map_00000.png,4KTN9
map_00001.png,7UTUP
```

Dùng web tool để gán nhãn nhanh:

```bash
venv\Scripts\activate
python label_server.py
# Mở http://localhost:8080
```

### 2. Huấn luyện model

```bash
venv\Scripts\activate
python train.py --use-real-data
```

Model sẽ được lưu vào `./captcha_trocr_model/`

### 3. Dự đoán CAPTCHA

```bash
venv\Scripts\activate
python inference.py data/map_00050.png
# Output: Kết quả: AB3K7
```

```python
# Trong code Python
from inference import CaptchaSolver

solver = CaptchaSolver()
result = solver.solve_captcha("path/to/captcha.png")
print(result)  # "AB3K7"
```

### 4. (Tùy chọn) Tạo data giả

```bash
venv\Scripts\activate
python generate_data.py
```

## Tham số huấn luyện

| Tham số | master (GPU) | nogpu (CPU) |
|---|---|---|
| Model | trocr-base-printed | trocr-base-printed |
| Batch size | 32 | 4 |
| Learning rate | 2e-5 | 2e-5 |
| Epochs | 30 | 30 |
| Workers | 4 | 0 |
| FP16 | ✅ | ❌ |

## Mẹo

- Càng nhiều data gán nhãn → model càng chính xác
- Tối thiểu 200-300 ảnh để model bắt đầu học được
- Mục tiêu 400-500 ảnh để đạt CER < 0.1

## License

MIT
