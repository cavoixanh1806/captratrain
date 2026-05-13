# 🤖 Hệ thống AI Giải CAPTCHA với TrOCR

Fine-tuning mô hình `microsoft/trocr-small-printed` để nhận dạng ảnh CAPTCHA
(chữ và số bị méo, nghiêng, có đường kẻ nhiễu và chấm nhiễu).

---

## 📁 Cấu trúc dự án

```
captratrain/
├── data/                        ← Ảnh CAPTCHA thực của bạn (500 ảnh)
│   ├── map_00000.png
│   ├── map_00001.png
│   └── metadata.csv             ← Bạn cần tạo file này (xem Bước 3)
│
├── data/synthetic/              ← Data giả (tự động tạo bởi generate_data.py)
│   ├── train/
│   │   ├── captcha_00000.png
│   │   └── metadata.csv
│   └── val/
│       ├── captcha_00000.png
│       └── metadata.csv
│
├── captcha_trocr_model/         ← Model đã train (tự động tạo bởi train.py)
│
├── generate_data.py             ← Tạo dataset CAPTCHA giả
├── dataset.py                   ← Tiền xử lý dữ liệu
├── train.py                     ← Huấn luyện model
├── inference.py                 ← Dự đoán CAPTCHA
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

### Bước 1: Tạo và kích hoạt môi trường ảo

```cmd
cd C:\Users\Administrator\Desktop\captratrain

python -m venv venv
venv\Scripts\activate
```

### Bước 2: Cài đặt dependencies

```cmd
pip install -r requirements.txt
```

> **Lưu ý GPU:** Nếu bạn có GPU NVIDIA, cài PyTorch với CUDA để train nhanh hơn:
> ```cmd
> pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🏷️ Bước 3: Dán nhãn data thực (QUAN TRỌNG)

Bạn đã có **500 ảnh CAPTCHA** trong thư mục `data/`. Cần tạo file `data/metadata.csv`:

### Định dạng metadata.csv

```csv
filename,text
map_00000.png,AB3K7
map_00001.png,X9PQ2
map_00002.png,MN4RT
...
```

### Cách tạo nhanh

Mở Excel hoặc Google Sheets, tạo 2 cột:
- Cột A: `filename` — tên file ảnh (ví dụ: `map_00000.png`)
- Cột B: `text` — chuỗi ký tự trong ảnh (ví dụ: `AB3K7`)

Lưu thành file CSV tại `data/metadata.csv`.

---

## 🚀 Các bước chạy

### Tùy chọn A: Dùng data thực của bạn (khuyến nghị)

Sau khi đã dán nhãn xong `data/metadata.csv`:

```cmd
python train.py --use-real-data
```

### Tùy chọn B: Tạo data giả rồi train

```cmd
# Bước 1: Tạo 10,000 ảnh train + 2,000 ảnh val
python generate_data.py

# Bước 2: Bắt đầu huấn luyện
python train.py
```

### Tùy chọn C: Kết hợp data thực + data giả (tốt nhất)

```cmd
# Tạo data giả trước
python generate_data.py

# Train với cả hai nguồn data
python train.py --use-real-data --combine
```

---

## 🔍 Kiểm thử (Inference)

### Giải một ảnh CAPTCHA

```cmd
python inference.py data/map_00000.png
```

Output:
```
Kết quả: AB3K7
```

### Dùng trong code Python

```python
from inference import CaptchaSolver

# Khởi tạo solver (load model một lần)
solver = CaptchaSolver()

# Giải một ảnh
result = solver.solve_captcha("data/map_00000.png")
print(result)  # "AB3K7"

# Giải nhiều ảnh cùng lúc (nhanh hơn)
results = solver.solve_batch([
    "data/map_00000.png",
    "data/map_00001.png",
    "data/map_00002.png",
])
print(results)  # ["AB3K7", "X9PQ2", "MN4RT"]
```

---

## 📊 Theo dõi quá trình train

Trong quá trình train, bạn sẽ thấy output như sau:

```
{'loss': 2.345, 'learning_rate': 4.5e-05, 'epoch': 1.0}
{'eval_loss': 1.234, 'eval_cer': 0.45, 'epoch': 1.0}
{'loss': 1.123, 'learning_rate': 3.0e-05, 'epoch': 3.0}
{'eval_loss': 0.567, 'eval_cer': 0.12, 'epoch': 3.0}
```

- **loss**: Loss trên tập train (càng thấp càng tốt)
- **eval_cer**: Character Error Rate trên tập val (mục tiêu < 0.05)

---

## 💡 Mẹo tối ưu

| Tình huống | Giải pháp |
|---|---|
| GPU bị OOM (hết bộ nhớ) | Giảm `BATCH_SIZE = 4` trong `train.py` |
| Train quá chậm trên CPU | Giảm `NUM_EPOCHS = 5`, `TRAIN_COUNT = 3000` |
| CER vẫn cao sau 10 epochs | Tăng `NUM_EPOCHS = 20`, giảm `LEARNING_RATE = 2e-5` |
| Muốn model chính xác hơn | Dùng `microsoft/trocr-base-printed` thay vì `small` |

---

## 🔧 Yêu cầu hệ thống

- Python 3.10+
- RAM: tối thiểu 8GB
- GPU: NVIDIA với CUDA (khuyến nghị, không bắt buộc)
- Dung lượng ổ cứng: ~5GB (model + data)
