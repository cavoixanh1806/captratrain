# 🤖 CAPTCHA Solver — Fine-tuning TrOCR

Hệ thống AI Local giải CAPTCHA bằng phương pháp Fine-tuning mô hình TrOCR (`microsoft/trocr-base-printed`) của Microsoft.

## Đặc điểm CAPTCHA hỗ trợ

- Chữ và số (A-Z, 0-9), 5 ký tự
- Chữ dính vào nhau, méo mó, nghiêng ngả
- Background có màu (không phải nền trắng)
- Đường kẻ nhiễu, chấm nhiễu
- Mỗi chữ một màu khác nhau

## Branches

| Branch | Dành cho | Batch size | GPU |
|---|---|---|---|
| `master` | CPU (Xeon E3, i5 cũ...) | 4 | Không cần |
| `gpu` | GPU NVIDIA (RTX 3060+) | 32 | Bắt buộc |

## Yêu cầu hệ thống

- Python 3.10+
- RAM: tối thiểu 8GB (khuyến nghị 12GB+)
- GPU NVIDIA (tùy chọn, không bắt buộc — có thể train trên CPU)

## Cài đặt

```bash
# Clone repo
git clone https://github.com/cavoixanh1806/captratrain.git
cd captratrain

# ── Chọn branch phù hợp với máy ──────────────────────────
# Máy CPU (mặc định, không cần làm gì thêm):
#   master branch — batch_size=4, không cần GPU

# Máy GPU (i5-12400F + RTX 3060 hoặc tương đương):
git checkout gpu
# ─────────────────────────────────────────────────────────

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

## Cấu trúc dự án

```
captratrain/
├── data/                    ← Ảnh CAPTCHA + metadata.csv (nhãn)
├── generate_data.py         ← Tạo data CAPTCHA giả (synthetic)
├── dataset.py               ← Tiền xử lý data cho model
├── train.py                 ← Huấn luyện (fine-tune) model
├── inference.py             ← Dự đoán CAPTCHA từ ảnh
├── label_server.py          ← Web tool gán nhãn nhanh
├── requirements.txt
└── README.md
```

## Sử dụng

### 1. Gán nhãn data

Ảnh CAPTCHA nằm trong `data/`. Cần tạo file `data/metadata.csv` với format:

```csv
filename,text
map_00000.png,4KTN9
map_00001.png,7UTUP
...
```

Dùng web tool để gán nhãn nhanh:

```bash
python label_server.py
# Mở http://localhost:8080
```

### 2. Huấn luyện model

```bash
# Train với data thực (sau khi đã gán nhãn)
python train.py --use-real-data
```

Model sẽ được lưu vào `./captcha_trocr_model/`

### 3. Dự đoán CAPTCHA

```bash
# Từ command line
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

### 4. (Tùy chọn) Tạo data giả để bổ sung

```bash
python generate_data.py
```

## Tham số huấn luyện

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Model | trocr-base-printed | Có thể đổi sang small/large |
| Batch size | 4 | Tăng nếu có nhiều RAM |
| Learning rate | 2e-5 | |
| Epochs | 30 | Có early stopping |
| Max length | 8 | 5 ký tự + special tokens |

## Mẹo

- Càng nhiều data gán nhãn → model càng chính xác
- Tối thiểu 200-300 ảnh để model bắt đầu học được
- Mục tiêu 400-500 ảnh để đạt CER < 0.1
- Nếu có GPU NVIDIA, train nhanh hơn 10-20x

## License

MIT
