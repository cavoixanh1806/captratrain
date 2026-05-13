# 🤖 CAPTCHA Solver — Fine-tuning TrOCR

Hệ thống AI Local giải CAPTCHA bằng phương pháp Fine-tuning mô hình TrOCR (`microsoft/trocr-base-printed`) của Microsoft.

## Branches

| Branch | Dành cho | GPU | Batch size | Tốc độ train |
|---|---|---|---|---|
| `master` | GPU — i5-12400F + RTX 3060+ | ✅ Bắt buộc | 32 | ~1-2 giờ |
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
- GPU: NVIDIA RTX 3060+ với CUDA 12.8+
- VRAM: tối thiểu 8GB

### Branch `nogpu` (CPU)
- Python 3.10+
- RAM: tối thiểu 8GB

## Cài đặt

### Cài Git (nếu chưa có)

```bash
# Windows — tải và cài từ:
# https://git-scm.com/download/win
# Hoặc dùng winget:
winget install Git.Git
```

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

## Giải thích output khi train

Khi train, bạn sẽ thấy output dạng:

```
{'loss': 3.29, 'grad_norm': 47.29, 'learning_rate': 5e-05, 'epoch': 4.44}
{'eval_loss': 3.47, 'eval_cer': 0.92, 'eval_runtime': 66.9, 'epoch': 4.44}
```

### Ý nghĩa các giá trị

| Giá trị | Ý nghĩa dễ hiểu | Tốt khi nào? |
|---|---|---|
| `loss` | Model sai bao nhiêu khi train. Như điểm "sai" | Càng nhỏ càng tốt (giảm dần) |
| `eval_loss` | Model sai bao nhiêu trên ảnh chưa thấy bao giờ | Càng nhỏ càng tốt |
| `eval_cer` | **Quan trọng nhất.** Tỷ lệ ký tự sai. VD: 0.64 = sai 64% ký tự | Mục tiêu < 0.1 (sai dưới 10%) |
| `learning_rate` | Tốc độ học — giảm dần theo thời gian | Tự động, không cần quan tâm |
| `grad_norm` | Độ lớn gradient — model đang thay đổi mạnh hay nhẹ | Tự động |
| `epoch` | Đã lặp qua toàn bộ data bao nhiêu lần | Chỉ để theo dõi tiến trình |

### Ví dụ thực tế với CER

- CER = 1.0 → sai hết (đoán "XXXXX" thay vì "4KTN9")
- CER = 0.64 → sai 3/5 ký tự (đoán "4KAAA" thay vì "4KTN9")
- CER = 0.2 → sai 1/5 ký tự (đoán "4KTN7" thay vì "4KTN9")
- CER = 0 → đúng hoàn toàn

### Tham số ảnh hưởng độ chính xác

| Tham số | Ảnh hưởng? | Giải thích |
|---|---|---|
| `NUM_EPOCHS` | ✅ Có | Model được học nhiều lần hơn → chính xác hơn (đến 1 mức nào đó) |
| `EVAL_STEPS` | ❌ Không | Chỉ là tần suất kiểm tra, không thay đổi cách model học |
| `BATCH_SIZE` | ⚠️ Ít | Ảnh hưởng tốc độ, ít ảnh hưởng kết quả cuối |
| `LEARNING_RATE` | ✅ Có | Quá cao → học sai, quá thấp → học chậm |
| **Số lượng data** | ✅✅✅ Quan trọng nhất | Nhiều data = chính xác hơn |

## Cập nhật code mới

Khi có code mới trên GitHub, chạy:

```bash
# GPU (branch master)
git pull origin master

# CPU (branch nogpu)
git pull origin nogpu
```

## License

MIT
