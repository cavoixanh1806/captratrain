# 🤖 CAPTCHA Solver — CRNN+CTC Pipeline

Hệ thống AI Local giải Minecraft Map CAPTCHA (128×128, 5 ký tự cố định) với
mục tiêu **exact_match ≥ 90%**, **CER ≤ 10%**.

Theo nghiên cứu trong [`research_minecraft_map_captcha_20260515.md`](research_minecraft_map_captcha_20260515.md):
**CRNN+CTC** là kiến trúc SOTA classic cho fixed-charset captcha có ký tự
chồng đè. Đây là replacement cho pipeline TrOCR cũ (đã loại bỏ vì
sub-word tokenizer không phù hợp + 334M params overfit nặng với dataset
nhỏ).

```
CAPTCHA 128×128 RGB
       │
       ▼
  ┌────────────────┐
  │ Resize 64×320  │  (kéo dãn ngang 1:5, đủ T=80 timesteps cho 5 chars)
  └────────┬───────┘
           ▼
  ┌────────────────────────────────────────────────────────┐
  │ CRNN backbone                                          │
  │   CNN 7 blocks (64→128→256→256→512→512→512)             │
  │     → (B, 512, h≈3, w=79)                                │
  │   AdaptivePool + reshape → (T=79, B, 512)                │
  │   BiLSTM 2 layers, hidden=256 → (T=79, B, 512)           │
  │   Linear → (T=79, B, NUM_CLASSES=25)                     │
  │   Canonical: count_parameters() == 8_718_937             │
  └────────┬───────────────────────────────────────────────┘
           ▼
  ┌────────────────┐
  │ CTC decoder    │  Greedy: argmax → collapse repeats → drop blanks
  └────────┬───────┘
           ▼
        "4KTN9"
```

## Charset (24 lớp + 1 blank cho CTC)

```
Chữ: A C D E F H J K L M N P Q R T U V W X Y  (20)
Số:  3 4 7 9                                  (4)
```

Loại các cặp dễ nhầm: `O/0`, `I/1`, `S/5`, `B/8`, `G/6`, `Z/2`.

## Yêu cầu hệ thống

- Python 3.10+, RAM 8GB
- GPU khuyến nghị: NVIDIA RTX 3060+ 8GB VRAM (CUDA 12.8+) — train ~30-45 phút
  cho 200 epoch trên ~500 ảnh real
- CPU only cũng chạy được — train ~10-15 giờ (chỉ smoke test)

## Cài đặt

```bash
git clone https://github.com/cavoixanh1806/captratrain.git
cd captratrain
setup.bat                       # tu dong tao venv + cai PyTorch CUDA 12.8 + requirements
```

Hoặc thủ công:
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements-dev.txt    # cho ai cần chạy test
```

## Chạy 1 lệnh

```bash
run_all.bat                          REM clean train, default 200 epochs
run_all.bat --epochs 100             REM override epoch budget
run_all.bat --batch-size 64          REM override batch size
run_all.bat --num-workers 4          REM override DataLoader workers (auto neu khong truyen)
run_all.bat --resume                 REM tiep tuc tu captcha_crnn_last.pth
run_all.bat --resume --epochs 50     REM resume + chinh epoch budget
```

Workflow tự động:
1. Import ảnh mới từ `dataset/` (idempotent, check label trùng).
2. Clean cũ (chỉ khi không có `--resume`).
3. Train CRNN+CTC trên `data/metadata.csv` (val 15% real, augment ON).
4. Eval trên toàn bộ real, in verdict.
5. Archive artifact vào `runs\run_<TS>\`.

Tất cả flag không liên quan tới `--resume` được forward thẳng xuống
`train_crnn.py` (xem `train_crnn.py --help` để biết thêm).

## Output sau khi chạy xong

### Tại repo root (`inference_crnn.py` / `eval_crnn.py` / Solver tìm tới đây)
- `captcha_crnn_model.pth` — best model (theo `val_exact_match`)
- `captcha_crnn_last.pth` — checkpoint epoch cuối, dùng cho `--resume`
- `captcha_crnn_model.onnx` — best model export ONNX (opset 14, dynamic batch)
- `train_log.txt` — log đầy đủ của lần chạy gần nhất

### Tại thư mục archive `runs\run_<YYYYMMDD_HHMMSS>\`
Mỗi lần chạy `run_all.bat` tạo một thư mục mới — không đè lên lần trước.

| File | Mô tả |
|---|---|
| `train_log.txt` | Log đầy đủ + verdict eval (text) |
| `metrics.csv` | Per-epoch table — đọc bằng pandas/Excel để vẽ chart |
| `eval_summary.json` | Eval kết quả structured: confusion matrix, low-confidence wrongs, ... |
| `captcha_crnn_model.pth` | Snapshot best model |
| `captcha_crnn_last.pth` | Snapshot last (cho resume sau này) |
| `captcha_crnn_model.onnx` | Snapshot ONNX |

`metrics.csv` columns: `epoch, timestamp, train_loss, train_grad_norm,
learning_rate, eval_loss, eval_cer, eval_exact_match, eval_runtime_s,
gap, is_best, best_val_em, best_epoch`.

`eval_summary.json` keys: `total, exact_match, cer, per_position,
confusions, confusion_matrix, avg_confidence, low_confidence_wrongs,
timestamp, checkpoint, metadata_path`.

`runs/` đã có trong `.gitignore` — không lo accidentally commit model nặng.

### Đọc metrics CSV trong Python
```python
import pandas as pd
df = pd.read_csv("runs/run_20260516_223000/metrics.csv")
df.plot(x="epoch", y=["train_loss", "eval_loss", "eval_exact_match"])
```

### Đọc confusion matrix JSON
```python
import json, pandas as pd, seaborn as sns
data = json.load(open("runs/run_20260516_223000/eval_summary.json"))
cm = pd.DataFrame(data["confusion_matrix"]).fillna(0).astype(int)
sns.heatmap(cm, annot=True, fmt="d")
```

## Đọc output khi train

Log được format kiểu Hugging Face Trainer (qua `tqdm.write` + `print`):

```
{'loss': '3.5138', 'grad_norm': '12.345', 'learning_rate': '5.000e-04', 'epoch': '10'}
{'eval_loss': '3.426', 'eval_cer': '0.945', 'eval_exact_match': '0.000', 'eval_runtime': '1.23', 'eval_samples_per_second': '91.870', 'eval_steps_per_second': '3.252', 'epoch': '10'}
```

- `loss` / `eval_loss` — CTC loss (càng thấp càng tốt)
- `eval_exact_match` — % captcha decode đúng cả 5/5 ký tự (metric chính)
- `eval_cer` — character error rate (0 = perfect, 1 = sai hết)
- `grad_norm` — norm gradient sau clip
- `learning_rate` — LR hiện tại sau scheduler step
- `eval_samples_per_second` / `eval_steps_per_second` — throughput

Khi best model được save:
```
2026-05-15 23:15:12 [INFO]   → Best model saved (val_exact_match=0.3540, val_cer=0.142)
```

Cuối training:
```
2026-05-15 23:38:00 [INFO] [DONE] Best val_exact_match=0.4071 at epoch 178
2026-05-15 23:38:00 [INFO]   Checkpoint: captcha_crnn_model.pth
2026-05-15 23:38:01 [INFO]   ONNX exported: captcha_crnn_model.onnx
```

### Eval verdict format

```
================================================================
CRNN EVALUATION RESULTS
================================================================
Total samples:       500
Exact match correct: 178
Exact match acc:      35.60%
CER:                  14.32%
Avg confidence:       62.40%

Per-position accuracy:
  Position 1:  92.40%  ████████████████████████████
  Position 2:  88.20%  ██████████████████████████
  Position 3:  79.60%  ███████████████████████
  ...

Top 10 confusions (322 total mistakes):
  3 → 7: 12
  K → X: 9
  Y → V: 7
  ...

================================================================
VERDICT
================================================================
[GOOD] Exact match 35.60% — close to target, fine-tune more.
```

## Thêm data mới

Đặt ảnh có label vào `dataset/` với format: `map_<LABEL>.png`

Ví dụ: `map_4KTN9.png`, `map_WTVRY.png`

Khi chạy `run_all.bat`, script tự import vào `data/` + cập nhật
`metadata.csv`. Ảnh trùng label sẽ skip.

## Chạy từng bước

```bash
venv\Scripts\activate

# Gan nhan (neu chua co metadata.csv) — Web UI tai localhost:8080
python label_server.py

# Import data moi tu dataset/
python import_new_data.py

# Train CRNN — co the truyen tat ca cac flag
python train_crnn.py
python train_crnn.py --epochs 50 --batch-size 64
python train_crnn.py --resume
python train_crnn.py --num-workers 4 --metrics-csv runs\my_run\metrics.csv

# Sinh synthetic data (mac dinh 100K) — KHONG bat boi default workflow
python generate_synthetic_crnn.py
python generate_synthetic_crnn.py --count 50000

# Train co synthetic
python train_crnn.py --use-synthetic

# Self-training round 2 (chay sau khi co round 1 tot)
python self_train.py

# Eval — co the dump JSON
python eval_crnn.py
python eval_crnn.py --batch-size 128 --json-out runs\my_run\eval_summary.json
```

## Inference

```python
from inference_crnn import CRNNCaptchaSolver

solver = CRNNCaptchaSolver()
text = solver.solve("path/to/captcha.png")
print(text)  # "4KTN9"

# Voi confidence
text, conf = solver.solve_with_confidence("path/to/captcha.png")
print(f"{text} ({conf:.2%})")

# Batch
results = solver.solve_batch(["a.png", "b.png", "c.png"])
```

```bash
python inference_crnn.py data/map_00001.png
```

### Deploy với ONNX runtime

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("captcha_crnn_model.onnx")
img = cv2.imread("captcha.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (320, 64)).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = np.transpose(img, (2, 0, 1))[None]                       # (1, 3, 64, 320)
logits = session.run(None, {"input": img.astype(np.float32)})[0]
# CTC decode: argmax + collapse + drop blank
# (xem inference_crnn.decode_greedy de biet logic)
```

## Cấu hình hyperparams (canonical, source of truth)

| Tham số | Giá trị | Nơi định nghĩa |
|---|---|---|
| Backbone | CNN 7 blocks (64→128→256→256→512→512→512) + BiLSTM 2-layer (hidden=256) | `crnn_model.CRNN.__init__` |
| Params | 8,718,937 (~8.72M) | `CRNN().count_parameters()` |
| Loss | CTCLoss (blank=0, zero_infinity=True) | `train_crnn.main` |
| Optimizer | AdamW (weight_decay=1e-4) | `train_crnn.main` |
| Default LR | 3e-4 | `train_crnn.DEFAULT_LR` |
| LR schedule | Linear warmup ≥ 2 epochs (`max(WARMUP_STEPS, steps_per_epoch * 2)`) → cosine decay → `lr × 0.01` | `train_crnn.build_warmup_cosine_scheduler` |
| Default epochs | 200 | `train_crnn.DEFAULT_EPOCHS` |
| Default batch | 64 (RTX 3060 8GB OK) | `train_crnn.DEFAULT_BATCH_SIZE` |
| Input size | 64×320 (resize từ 128×128, ratio 1:5) | `crnn_model.INPUT_HEIGHT/WIDTH` |
| Augment | Albumentations toned-down: Affine ±4°, Perspective 0.01–0.04, ColorJitter mild, GaussNoise 3–12, OneOf(GaussianBlur/MotionBlur), CoarseDropout 2 holes 5×5 | `dataset_crnn._build_albu_aug(strong=True)` |
| Mixed precision | FP16 (CUDA only) | `train_crnn.main` |
| Gradient clip | 5.0 | `train_crnn.GRAD_CLIP_NORM` |
| Val split | 15% real, `seed=42` | `dataset_crnn.create_crnn_datasets` |
| DataLoader workers | Auto: Windows `min(4, cpu_count // 2)`; Linux/macOS `min(8, cpu_count // 2)`; falls back to `0` on torchvision path. Override with `--num-workers`. Khi `> 0`: `persistent_workers=True`, `prefetch_factor=4` | `train_crnn._auto_num_workers` |

**Thời gian ước tính (200 epoch, ~500 ảnh real):**

| Hardware | Thời gian | Verdict |
|---|---|---|
| RTX 5090 | ~10-15 phút | Optimal |
| RTX 3060 8GB | **~30-45 phút** | Recommended |
| CPU-only (8 threads, 12GB RAM) | **~10-15 giờ** ⚠️ | Smoke test only |

> Chạy `python system_info.py` để check máy bạn được verdict gì.

## Tăng accuracy nếu < 90%

Pipeline mặc định (chỉ real, không synthetic, không self-train) là **baseline tối giản**. Nếu accuracy < 90%, bật từng kỹ thuật theo thứ tự đề xuất:

1. **Bật synthetic data** (khả năng cải thiện cao nhất):
   ```bash
   python generate_synthetic_crnn.py --count 100000
   python train_crnn.py --use-synthetic
   python eval_crnn.py
   ```

2. **Self-training round 2** (sau khi có round 1 tốt ≥ 70%):
   ```bash
   python self_train.py
   python eval_crnn.py
   ```

3. **Train lâu hơn**:
   ```bash
   python train_crnn.py --epochs 400
   ```

4. **Tăng synthetic count** lên 200K:
   ```bash
   python generate_synthetic_crnn.py --count 200000
   ```

5. **Calibrate `synthetic_renderer.py`** — match font/color/overlap với real
   nếu domain gap synthetic vs real quá lớn.

## Testing (dev)

Spec [`crnn-ctc-collapse-fix`](.kiro/specs/crnn-ctc-collapse-fix/) đi kèm
2 bộ property test:

- `tests/test_property1_bug_condition.py` — Property 1: post-fix model
  generalises to real val + GPU ≥ 80% util. Smoke (20 epoch) + full
  (200 epoch, `-m slow`) + GPU util sub-tests gate trên hardware target.
  Helper checks: replay log + doc-drift grep.
- `tests/test_property2_preservation.py` — Property 2: 10 invariants không
  được đổi (charset, input shape, ONNX contract, resume flow, decode
  semantics, split determinism, Solver API, ...).

```bash
venv\Scripts\python.exe -m pytest tests\ -v -m "not slow"     # quick
venv\Scripts\python.exe -m pytest tests\ -v -m slow            # full 200-epoch (chi chay tren RTX 3060)
```

Hardware-gated tests skip cleanly trên CPU host với reason rõ ràng.

## Cấu trúc project

```
captratrain/
├── data/                           # ~500 real images + metadata.csv
├── dataset/                        # Inbox cho ảnh mới chờ import
├── runs/                           # Per-run archives (gitignored)
├── docs/
│   ├── adr/0001-crnn-ctc-over-softmax.md
│   ├── codebase/                   # ARCHITECTURE, CONCERNS, STACK, ...
│   └── research_strategy_20260515.md
├── tests/                          # Property 1 + Property 2 test suites
├── .kiro/                          # Spec + skills (Kiro workflow files)
├── crnn_model.py                   # CRNN architecture + CTC encode/decode
├── dataset_crnn.py                 # CRNNCaptchaDataset + augmentation
├── synthetic_renderer.py           # render_text_on_image (calibrated)
├── generate_synthetic_crnn.py      # Sinh synthetic data (default 100K)
├── train_crnn.py                   # Train CTC + warmup-cosine + metrics CSV
├── self_train.py                   # Self-training round 2
├── inference_crnn.py               # CRNNCaptchaSolver (CLI + lib)
├── eval_crnn.py                    # Evaluation suite + JSON dump
├── import_new_data.py              # Import ảnh mới từ dataset/
├── label_server.py                 # Web UI gán nhãn
├── system_info.py                  # Kiểm tra cấu hình máy
├── run_all.bat                     # Pipeline runner (resume-aware, archives)
├── run_smoke.bat                   # Smoke test (5 epochs, batch 16)
├── setup.bat                       # Auto setup venv + PyTorch + deps
├── pytest.ini
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # pytest + hypothesis cho dev
└── train_log.txt                   # Log lan chay gan nhat (root mirror)
```

## Cập nhật code

```bash
git pull origin master
```

## License

MIT
