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

Mở CMD trước (không double-click .bat), rồi chạy:

```bash
cmd /k "venv\Scripts\activate & run_all.bat"
```

Hoặc nếu đã mở CMD sẵn:

```bash
venv\Scripts\activate
run_all.bat
```

Workflow tự động chạy 5 bước:
1. Extract 754 real backgrounds (inpainting xóa text)
2. Generate 12K U-Net pairs (BG thật + text mới)
3. Generate 6K TrOCR synthetic samples (có label biết trước)
4. Train U-Net denoiser (~10-15 phút)
5. Train TrOCR với Combine synthetic+real (~60-90 phút)
6. Evaluate trên 754 ảnh real → in kết quả

## Import data mới

Nếu có thêm ảnh mới (format `map_<LABEL>.png` trong thư mục `dataset/`):
```bash
python import_new_data.py
```
Tự động đổi tên + append vào `data/metadata.csv`.

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
python eval_model.py
```

## Inference (sau khi train xong)

```python
from inference import CaptchaSolver

solver = CaptchaSolver(model_dir="./captcha_trocr_model")
result = solver.solve_captcha("data/map_00001.png")
print(result)  # "4KTN9"

# Batch — nhanh hơn khi giải nhiều ảnh
results = solver.solve_batch(["img1.png", "img2.png"])
```

```bash
python inference.py data/map_00001.png
```

---

## Giải thích log khi train

### Bước 2 — Extract backgrounds

```
Tim thay 500 anh. Bat dau extract backgrounds...
  50/500 — processed=50, failed=0
  100/500 — processed=100, failed=0
  ...
[DONE] Processed 500/500 backgrounds
```

| Log | Ý nghĩa |
|-----|---------|
| `processed=500` | Số ảnh đã xóa text thành công |
| `failed=0` | Số ảnh bị lỗi đọc file (bình thường = 0) |

---

### Bước 3 — Generate U-Net data

```
Loaded 500 real backgrounds vao cache
Generating 20000 pairs for [train]...
  [train] 2000/20000 pairs done
  [train] 4000/20000 pairs done
  ...
  [train] Done: 20000 pairs saved
Generating 4000 pairs for [val]...
  [val] Done: 4000 pairs saved
[DONE] U-Net training data saved to: data/unet_pairs
```

| Log | Ý nghĩa |
|-----|---------|
| `Loaded 500 real backgrounds` | Đã load BG thật vào RAM |
| `20000 pairs` | Mỗi pair = 1 ảnh noisy + 1 mask (text=trắng, bg=đen) |
| `train/val` | 20K train + 4K validation |

---

### Bước 4 — Train U-Net

```
Device: cuda
GPU: NVIDIA GeForce RTX 3060
Train: 10000 samples, 312 batches
Val:   2000 samples, 62 batches
Model params: 7,763,041
Training for 30 epochs...
--------------------------------------------------------------------------------
Epoch   1/30 (45.2s) | Train Loss: 0.4521, IoU: 0.3214 | Val Loss: 0.3812, IoU: 0.4523, Acc: 0.8234
  -> Best model saved (IoU: 0.4523)
Epoch   2/30 (44.8s) | Train Loss: 0.2134, IoU: 0.5678 | Val Loss: 0.1923, IoU: 0.6234, Acc: 0.9012
  -> Best model saved (IoU: 0.6234)
...
Epoch  25/30 (44.5s) | Train Loss: 0.0312, IoU: 0.8912 | Val Loss: 0.0398, IoU: 0.8756, Acc: 0.9823
  -> Best model saved (IoU: 0.8756)
Epoch  30/30 (44.3s) | Train Loss: 0.0289, IoU: 0.9012 | Val Loss: 0.0412, IoU: 0.8734, Acc: 0.9812
--------------------------------------------------------------------------------
[DONE] Best val IoU: 0.8756 at epoch 25
Model saved to: captcha_unet_model.pth
```

| Log | Ý nghĩa | Tốt khi |
|-----|---------|---------|
| `Train Loss` | Sai số DiceBCE trên tập train — model đang học | Giảm dần |
| `Val Loss` | Sai số trên ảnh chưa thấy — đo generalization | Giảm dần, gần Train Loss |
| `IoU` | % vùng chữ tách đúng (Intersection over Union) | **> 0.85** |
| `Acc` | % pixel phân loại đúng (text/background) | > 0.95 |
| `Best model saved` | Lưu model tốt nhất (IoU cao nhất trên val) | Tự động |

**Cách đọc:**
- IoU tăng dần từ 0.32 → 0.87 = model đang học tốt
- Val IoU > 0.85 = U-Net đã tách chữ chính xác
- Nếu Val Loss tăng trong khi Train Loss giảm = overfitting (cần thêm data)

---

### Bước 5 — Train TrOCR

```
🚀 Sử dụng GPU: NVIDIA GeForce RTX 3060
Đang load TrOCRProcessor từ: microsoft/trocr-base-printed
Mode: Real Data + Preprocessing(unet) + Augmentation
Train samples: 400
Val samples:   100
✅ Model đã được cấu hình thành công.
🏋️  Bắt đầu fine-tuning...
{'loss': 3.29, 'learning_rate': 5e-05, 'epoch': 4.0}
{'eval_loss': 2.81, 'eval_cer': 0.85, 'eval_exact_match': 0.02, 'epoch': 4.0}
{'loss': 1.52, 'learning_rate': 4.5e-05, 'epoch': 8.0}
{'eval_loss': 1.23, 'eval_cer': 0.42, 'eval_exact_match': 0.18, 'epoch': 8.0}
{'loss': 0.68, 'learning_rate': 3.8e-05, 'epoch': 16.0}
{'eval_loss': 0.54, 'eval_cer': 0.15, 'eval_exact_match': 0.62, 'epoch': 16.0}
{'loss': 0.31, 'learning_rate': 2.5e-05, 'epoch': 28.0}
{'eval_loss': 0.28, 'eval_cer': 0.06, 'eval_exact_match': 0.88, 'epoch': 28.0}
{'loss': 0.18, 'learning_rate': 1.2e-05, 'epoch': 40.0}
{'eval_loss': 0.21, 'eval_cer': 0.04, 'eval_exact_match': 0.92, 'epoch': 40.0}
[SAVE] Best model saved to: ./captcha_trocr_model
[DONE] Training complete!
```

| Log | Ý nghĩa | Tốt khi |
|-----|---------|---------|
| `loss` | Sai số trên tập train | Giảm dần (3.29 → 0.18) |
| `eval_loss` | Sai số trên 100 ảnh val | Giảm dần |
| `eval_cer` | % ký tự sai (Character Error Rate) | **< 0.10** (dưới 10%) |
| `eval_exact_match` | % ảnh đoán đúng hoàn toàn 5/5 ký tự | **≥ 0.90** (90%+) |
| `learning_rate` | Tốc độ học — tự giảm dần | Không cần quan tâm |
| `epoch` | Số lần lặp qua toàn bộ 400 ảnh train | Chỉ để theo dõi |

**Cách đọc:**
- `eval_exact_match` tăng dần: 0.02 → 0.18 → 0.62 → 0.88 → 0.92
- Khi đạt 0.90+ = model đã đạt mục tiêu
- Training tự dừng nếu `exact_match` không cải thiện sau 8 lần eval liên tiếp

**Ví dụ thực tế:**
```
eval_cer: 0.85 → sai gần hết (mới bắt đầu học)
eval_cer: 0.42 → sai 2/5 ký tự trung bình
eval_cer: 0.15 → sai ~1 ký tự / ảnh
eval_cer: 0.04 → gần hoàn hảo (sai 0.2 ký tự / ảnh)

eval_exact_match: 0.02 → chỉ 2% ảnh đúng hoàn toàn
eval_exact_match: 0.62 → 62% ảnh đúng
eval_exact_match: 0.92 → 92% ảnh đúng ← MỤC TIÊU ĐẠT
```

---

### Bước 6 — Evaluate

```
============================================================
EVALUATION RESULTS
============================================================
Total samples:       500
Exact match correct: 456
Exact match acc:     91.20%
CER:                 4.30%

Per-position accuracy:
  Position 1: 97.40%
  Position 2: 96.80%
  Position 3: 95.60%
  Position 4: 94.20%
  Position 5: 93.80%

Top 10 confusions (label → predicted):
  0 → O: 8
  1 → I: 5
  8 → B: 4
  5 → S: 3
  ...

============================================================
VERDICT
============================================================
[EXCELLENT] Exact match 91.2% >= 90% — DAT MUC TIEU!
```

| Log | Ý nghĩa |
|-----|---------|
| `Exact match acc: 91.20%` | 456/500 ảnh đoán đúng hoàn toàn 5/5 ký tự |
| `CER: 4.30%` | Trung bình sai 4.3% ký tự (0.2 ký tự / ảnh) |
| `Per-position accuracy` | Độ chính xác từng vị trí (1-5) — vị trí cuối thường kém hơn |
| `Top 10 confusions` | Những cặp ký tự hay bị nhầm (0↔O, 1↔I, 8↔B...) |
| `[EXCELLENT]` | Đạt mục tiêu 90%+ |

**Nếu chưa đạt 90%:**
- `[GOOD] 80-89%` → Tăng epochs hoặc thêm data
- `[OK] 50-79%` → Kiểm tra U-Net IoU, có thể BG extraction chưa tốt
- `[FAIL] <50%` → Kiểm tra metadata.csv có đúng label không

---

## Cấu hình hiện tại

### Phần cứng tối ưu
```
GPU:  NVIDIA RTX 3060 8GB VRAM
CPU:  Intel i5-12400F (6 cores / 12 threads)
RAM:  16GB
CUDA: 12.8+
```

### U-Net Denoiser
```
Model:          CaptchaUNet (7.7M params)
Input:          (3, 128, 128) RGB
Output:         (1, 128, 128) probability map
Loss:           DiceBCE (50% Dice + 50% BCE)
Optimizer:      Adam (lr=1e-3, weight_decay=1e-4)
Scheduler:      CosineAnnealingLR (eta_min=1e-6)
Batch size:     32
Epochs:         30
Train data:     10,000 pairs (real BG + synthetic text)
Val data:       2,000 pairs
Workers:        4
```

**Giải thích U-Net:**

| Tham số | Tại sao chọn giá trị này |
|---------|--------------------------|
| `DiceBCE Loss` | Text chỉ chiếm 15-20% ảnh → class imbalance. BCE bias về BG. Dice focus vào overlap → cân bằng |
| `lr=1e-3` | U-Net train from scratch → cần lr cao. TrOCR fine-tune nên lr thấp hơn (5e-5) |
| `weight_decay=1e-4` | Regularization nhẹ tránh overfitting trên synthetic data |
| `CosineAnnealingLR` | Đầu lr cao (học nhanh), cuối lr thấp (tinh chỉnh). Tốt hơn step decay |
| `Batch=32` | Ảnh 128×128 nhỏ, RTX 3060 chứa dư. Batch lớn = gradient ổn định |
| `30 epochs` | U-Net converge nhanh (task đơn giản: binary segmentation) |
| `10K+2K` | Đủ diversity, giảm tải GPU (nhiệt ~60°C thay vì 70°C) |

### TrOCR OCR
```
Base model:     microsoft/trocr-base-printed (~334M params)
Input:          Preprocessed image (U-Net output)
Output:         5 ký tự (A-Z, 0-9)
Loss:           CrossEntropy (auto từ Seq2SeqTrainer)
Optimizer:      AdamW (lr=5e-5, weight_decay=0.01)
Warmup:         100 steps (~8% total)
Batch size:     16
Epochs:         50 (EarlyStopping patience=8)
Train data:     400 real images (80% of 500)
Val data:       100 real images (20% of 500)
Beam search:    4 beams
MAX_LENGTH:     8 tokens ([BOS] + 5 chars + [EOS] + 1 spare)
FP16:           Enabled (mixed precision)
Workers:        4
Metric:         exact_match (best model saved by highest)
Augmentation:   ElasticTransform, GridDistortion, GaussNoise, ColorJitter, Blur
```

**Giải thích TrOCR:**

| Tham số | Tại sao chọn giá trị này |
|---------|--------------------------|
| `lr=5e-5` | Fine-tune model pretrained → lr cao phá weights đã học. 5e-5 là chuẩn transformer |
| `warmup=100` | Đầu training lr tăng dần 0→5e-5, tránh gradient explosion khi weights chưa ổn |
| `weight_decay=0.01` | Regularization mạnh hơn U-Net vì chỉ có 400 ảnh (ít data, dễ overfit) |
| `Batch=16` | TrOCR 334M params lớn hơn U-Net 7.7M → cần nhiều VRAM hơn → batch nhỏ hơn |
| `50 epochs + patience=8` | 400 ảnh ít → cần nhiều epochs. EarlyStopping dừng sớm nếu converge |
| `Beam=4` | Tìm 4 chuỗi xác suất cao nhất, chọn tốt nhất. >4 chậm mà không tăng accuracy |
| `MAX_LENGTH=8` | CAPTCHA 5 ký tự + [BOS] + [EOS] = 7. Dư 1 token an toàn |
| `FP16` | Giảm VRAM 50%, tăng tốc 30%. Không ảnh hưởng accuracy trên RTX 3060 |
| `exact_match` | CAPTCHA 1 ký tự sai = fail → chọn model theo exact_match, không phải CER |
| `Augmentation` | 400 ảnh quá ít → augmentation tạo biến thể mới mỗi epoch ≈ 2000-3000 ảnh |

### Data Generation (generate_unet_data.py)
```
Real BG source: data/real_backgrounds/ (500 extracted backgrounds)
BG augment:     flip horizontal 50%, brightness ±15
Text colors:    70% red/orange, 9% cyan/blue, 8% purple, 5% yellow
Font:           Mix bold (60%) + regular (40%), per-character random
Font size:      36-50px (capped at 50 to fit 128px canvas)
Char overlap:   55% dense, 25% medium, 20% light
Rotation:       65% ±5°, 21% ±15°, 11% ±30°, 3% ±44°
Noise lines:    40% none, 40% 1-3 lines, 20% 3-6 lines (BG already has lines)
Noise dots:     0-10 (very few)
Wave distort:   50% probability, amplitude 1-3px
```

---

## Cập nhật code mới

```bash
git pull origin master
```

## License

MIT
