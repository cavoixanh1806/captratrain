# 📋 Tóm tắt pipeline CRNN+CTC

## 🎯 Mục tiêu

- Train model giải Minecraft Map CAPTCHA (128×128, 5 ký tự, charset 24 chars)
- Target: **exact_match ≥ 90%, CER ≤ 10%**

## 🏗️ Kiến trúc đã chọn

```
CAPTCHA 128×128 RGB
       │
       ▼
  Resize 64×320 (ratio 1:5, kéo dãn ngang)
       │
       ▼
  CRNN: CNN 7-block + BiLSTM(128) × 2  (~2.18M params)
       │
       ▼
  Output (T=79, B, 25 classes: 24 chars + 1 CTC blank)
       │
       ▼
  CTC greedy decode → "4KTN9"
```

## 📂 Cấu trúc project

| File | Vai trò |
|---|---|
| `crnn_model.py` | CRNN architecture + CTC encode/decode/save/load/ONNX |
| `dataset_crnn.py` | Dataset + augmentation (Affine, ColorJitter, Noise, Blur, Cutout) + collate CTC |
| `train_crnn.py` | Train pipeline tối giản (200 epochs, AdamW + warmup-cosine + AMP fp16, **không EMA, không EarlyStopping**) |
| `eval_crnn.py` | Eval suite (exact_match, CER, per-position, confusions, confidence) |
| `inference_crnn.py` | CRNNCaptchaSolver (CLI + Python API) |
| `self_train.py` | Self-training round 2 (chạy riêng khi cần) |
| `generate_synthetic_crnn.py` | Sinh synthetic data 100K (chạy riêng khi cần) |
| `synthetic_renderer.py` | Render text trên BG calibrated từ 754 real |
| `system_info.py` | Check máy + verdict (READY/MARGINAL/CPU_ONLY/NOT_RECOMMENDED) |
| `run_smoke.bat` | **Smoke test nhanh** (5 epochs, batch 16, ~30-60 phút CPU) |
| `run_all.bat` | **Train chính** (200 epochs, batch 32, ~30-45 phút RTX 3060) |

## ⚙️ Hyperparams

| Tham số | Giá trị |
|---|---|
| Backbone | CNN 7 blocks + BiLSTM(256) × 2 |
| Params | 2.18M |
| Loss | CTCLoss (blank=0, zero_infinity=True) |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR | 1e-3 → linear warmup 200 → cosine → 1e-5 |
| Epochs | 50 (full, không early stop) |
| Batch | 64 (16 cho CPU smoke) |
| Input | 64×320 |
| Augment | RandomAffine ±5°, ColorJitter, GaussNoise, Blur, CoarseDropout |
| AMP | FP16 (CUDA only) |
| Grad clip | 5.0 |
| Val split | 15% real |

## 📊 Logging

Cả 2 script `.bat` sử dụng script Python inline (`sys.stdout.write` + `f.write`) để khắc phục triệt để lỗi mã hóa (binary/UTF-16 LE mixed ASCII) thường gặp của PowerShell `Tee-Object` trên Windows:

- ✅ **Vừa hiện ra CMD** (với thanh tiến trình `tqdm` kiểu Hugging Face Trainer)
- ✅ **Vừa ghi file log** (đảm bảo chuẩn `UTF-8` thuần túy, mở được trên mọi text editor)
  - `run_smoke.bat` → `smoke_log.txt`
  - `run_all.bat` → `train_log.txt`

Auto-detect `venv\Scripts\python.exe`, fallback về `python` trong PATH.

## 🚀 Cách chạy

### Bước 1: Smoke test trên máy chính (Xeon E3, no GPU)

```cmd
cd C:\Users\Administrator\Desktop\captratrain
run_smoke.bat
```

**Kỳ vọng:**

- Thời gian: ~30-60 phút
- Loss giảm từ ~14 xuống ~3-5
- Có prediction (dù sai), không crash
- Output 2 file: `captcha_crnn_model.pth` + `smoke_log.txt`

> **KHÔNG kỳ vọng đạt 90%** — chỉ verify code chạy đúng.

### Bước 2: Push lên git

```cmd
git add .
git commit -m "CRNN+CTC pipeline minimal version"
git push origin master
```

### Bước 3: Pull về máy training (RTX 3060+) và chạy

```cmd
cd <project-dir>
git pull origin master

REM Check máy:
venv\Scripts\python.exe system_info.py

REM Chạy full:
run_all.bat
```

**Kỳ vọng trên RTX 3060:**

- Thời gian: ~30-45 phút
- Loss giảm về ~0.1-0.5
- val_exact_match: 60-85% sau round 1
- Output: `captcha_crnn_model.pth` + `captcha_crnn_model.onnx` + `train_log.txt`

### Bước 4 (nếu accuracy < 90%, làm tuần tự)

```cmd
REM 4a. Bật synthetic data (cải thiện cao nhất)
venv\Scripts\python.exe generate_synthetic_crnn.py --count 100000
venv\Scripts\python.exe train_crnn.py --use-synthetic
venv\Scripts\python.exe eval_crnn.py

REM 4b. Self-training round 2 (sau khi round 1 đạt ≥ 70%)
venv\Scripts\python.exe self_train.py
venv\Scripts\python.exe eval_crnn.py

REM 4c. Train lâu hơn
venv\Scripts\python.exe train_crnn.py --epochs 100
```

## ⏱️ Thời gian ước tính

| Hardware | Smoke test | Full train | Full + synthetic | Full + self-train |
|---|---|---|---|---|
| RTX 5090 | ~2-3 phút | ~10-15 phút | ~20-30 phút | +5 phút |
| RTX 3060 8GB | ~5-10 phút | ~30-45 phút | ~60-90 phút | +10 phút |
| **CPU (Xeon E3)** | **~30-60 phút** ✅ | ~10-15h ⚠️ | ~50-100h ❌ | +1-2h |

> Trên máy bạn (CPU): **chỉ chạy `run_smoke.bat`**, KHÔNG chạy `run_all.bat`.

## ✅ Đã verified

- Tất cả 9 modules import OK
- CRNN forward pass: input `(B,3,64,320)` → output `(T=79,B,25)`
- Dataset 754 samples, encode đúng `'4KTN9' → [22,8,15,11,24]`
- Smoke train trước đó: loss 14.4 → 3.8 sau 2 epochs (proof CTC học được)

## 🎬 Next steps

Bạn chạy:

```cmd
cd C:\Users\Administrator\Desktop\captratrain
run_smoke.bat
```

Sau khi xong (~30-60 phút), gửi nội dung `smoke_log.txt` (vài trăm dòng cuối)
hoặc screenshot CMD. Kiểm tra:

1. Loss có giảm không?
2. Có prediction nào ra (dù sai) không?
3. Có crash giữa chừng không?

Nếu OK → push git → pull về máy training → chạy `run_all.bat`.

## 📝 Các tùy chọn

| Option | Mô tả | Khi nào chọn |
|---|---|---|
| **A** | Chạy `run_smoke.bat` ngay (background, ~30-60 phút), agent monitor + báo kết quả | Khi muốn tự động hoàn toàn |
| **B** | Tự chạy, gửi log lại | Khi muốn kiểm soát |
| **C** | Skip smoke test, push thẳng git | Khi tin code OK (đã pass diagnostics + import test) |
