# CRNN CTC Collapse Fix — Bugfix Design

## Overview

CRNN+CTC training trên 754 ảnh thật (`data/metadata.csv`) thất bại trên hai
trục đồng thời:

1. **CTC collapse + augmentation overfit** — `eval_exact_match = 0.000` ở
   mọi epoch (1 → 200), `eval_loss` đi từ `41.118` → `3.823`, **vượt** trần
   "predict toàn blank" `log(NUM_CLASSES) = log(25) ≈ 3.219`, trong khi
   `train_loss` giảm xuống `2.5765`. Gap `eval_loss − train_loss` mở rộng từ
   `+0.10` (e50) đến `+1.25` nats (e200) — overfit lên augmentation chứ
   không học mapping image → text.
2. **CPU-bound DataLoader** — `num_workers = 0` cứng cho Windows + toàn bộ
   albumentations chạy đồng bộ trong `__getitem__` → GPU util ~50%, 12 CPU
   threads bão hòa.

Chiến lược fix có 4 phần, tất cả độc lập về layer nhưng cộng hưởng:

- **A — Augmentation pipeline (dataset_crnn.py):** giảm độ mạnh và xác suất
  để gap distribution train/val không vượt khả năng CRNN+CTC học từ 641 ảnh
  train.
- **B — Hyperparam + DataLoader (train_crnn.py):** hạ base LR phù hợp với
  corpus nhỏ, tăng `num_workers` an toàn trên Windows (top-level pickleable
  augmentation pipeline + `persistent_workers=True` + `prefetch_factor`),
  thêm CLI flag, giữ `num_workers=0` là fallback.
- **C — Architecture canonicalization (crnn_model.py):** chốt
  `hidden_size = 128` (giá trị in `2,186,553` params trong run lỗi và là
  cấu hình đang thực sự chạy) là nguồn sự thật duy nhất, đồng thời sửa
  docstring nội bộ trong `crnn_model.py` đang nói nhầm "hidden=256".
- **D — Doc/code reconciliation:** cập nhật `README.md`,
  `PIPELINE_SUMMARY.md`, `CLAUDE.md`, `docs/adr/0001-...`,
  `docs/codebase/ARCHITECTURE.md`, `docs/codebase/CONCERNS.md` để hoặc
  tham chiếu hằng số theo tên hoặc quote đúng các giá trị canonical
  (`hidden_size=128`, `DEFAULT_EPOCHS=200`, `DEFAULT_BATCH_SIZE=32`,
  `count_parameters() == 2_186_553`).

Phạm vi: chỉ chạm `dataset_crnn.py`, `train_crnn.py`, `crnn_model.py` (về
code) và 6 file doc nói trên. **Không** sinh synthetic data, **không** đổi
hardware target, **không** đổi charset/input-shape/ONNX-contract/Solver-API.

## Glossary

- **Bug_Condition (C)** — Một training run của `train_crnn.main` được gọi
  trên `data/metadata.csv` (754 ảnh real, train=641/val=113, `seed=42`)
  với `(augmentation, learning_rate, num_workers)` ở giá trị **trước fix**
  (`_TRAIN_AUG = _build_albu_aug(strong=True)` hiện tại + `DEFAULT_LR=1e-3`
  + `num_workers=0` hard-code) trên hardware target Windows + i5-12400F +
  RTX 3060 8GB + CUDA 12.8.
- **Property (P)** — Run đã sửa trên cùng dataset/seed/hardware đạt cả 4
  ngưỡng: `max_epoch eval_exact_match ≥ 0.30` AND `min_epoch eval_loss <
  3.219` AND `min_epoch (eval_loss − train_loss) ≤ 0.50` AND GPU util ≥
  80% trung bình trong cửa sổ steady-state 30s.
- **Preservation** — Mọi behavior **không** liên quan tới hyperparam/
  augmentation/DataLoader-workers/comment-text giữ nguyên byte-for-byte:
  charset, input shape, ONNX contract, resume flow, decode semantics, val
  split logic, Solver API, "no synthetic data" stance, hardware envelope.
- **CTC blank collapse** — Hiện tượng CRNN+CTC predict gần-toàn-blank để
  tối thiểu hóa loss, biểu hiện qua `train_loss ≈ log(NUM_CLASSES)` mắc
  kẹt và `eval_exact_match = 0` xuyên suốt.
- **Augmentation overfit** — Mode trong đó train_loss tiếp tục giảm do
  model memorize augmentation patterns, trong khi eval_loss tăng vì val
  ảnh không có những distortion đó. Quan sát từ epoch 18 đến 200 trong
  run lỗi.
- **`_TRAIN_AUG`** — `albumentations.Compose` ở module-level của
  `dataset_crnn.py`, được khởi tạo bởi `_build_albu_aug(strong=True)`.
- **`_TV_TRAIN_AUG`** — fallback `torchvision.transforms.Compose` khi
  albumentations vắng mặt (giữ nguyên trong fix này).
- **Canonical hyperparams** — Bộ giá trị nguồn sự thật duy nhất trong code:
  `crnn_model.CRNN.__init__(hidden_size=128)`,
  `train_crnn.DEFAULT_EPOCHS=200`, `train_crnn.DEFAULT_BATCH_SIZE=32`,
  `INPUT_HEIGHT=64`, `INPUT_WIDTH=320`, `NUM_CLASSES=25`,
  `CTC_BLANK_INDEX=0`, `count_parameters() == 2_186_553`.
- **Hardware target** — Windows + i5-12400F (6P+0E, 12T) + 16GB DDR4 +
  RTX 3060 8GB (CUDA 12.8); VRAM peak ≤ 8GB, RAM peak ≤ 16GB ở
  `DEFAULT_BATCH_SIZE=32`.

## Bug Details

### Bug Condition

Bug xuất hiện khi `train_crnn.main` được chạy trên `data/metadata.csv`
(754 ảnh real, train/val 85/15, `seed=42`) với cấu hình code hiện tại của
`dataset_crnn._build_albu_aug(strong=True)` + `DEFAULT_LR=1e-3` +
`num_workers=0` trên hardware target. Bug biểu hiện đồng thời ở 3 chỉ số:
(a) `eval_exact_match` không bao giờ thoát `0.000`, (b) `eval_loss` mắc
kẹt rồi vượt `log(25)`, (c) GPU util trung bình ~50% trong khi 12 CPU
threads ở 100%.

**Formal Specification:**

```
FUNCTION isBugCondition(input)
  INPUT:  input = TrainRun {
            dataset:         "data/metadata.csv" (754 real captchas),
            train_size:      641,
            val_size:        113,
            seed:            42,
            augmentation:    _TRAIN_AUG (strong albumentations pipeline),
            base_lr:         1e-3,
            warmup_steps:    200,
            epochs:          200,
            batch_size:      32,
            num_workers:     0,
            hardware:        {OS: Windows, CPU: i5-12400F, GPU: RTX 3060 8GB, CUDA 12.8}
          }
  OUTPUT: boolean

  // Run is buggy if it triggers CTC collapse + overfit AND/OR DataLoader
  // is the throughput bottleneck.
  collapse_or_overfit :=
        (max over epochs of eval_exact_match  == 0.000)
     OR (min over epochs of eval_loss         >= log(NUM_CLASSES))   // log(25) ≈ 3.219
     OR (min over epochs of (eval_loss - train_loss) > 0.50)

  cpu_bound :=
        (mean GPU utilization over 30s steady-state window < 80%)
    AND (DataLoader is the bottleneck on hardware target)

  doc_drift :=
        (any doc among {README.md, PIPELINE_SUMMARY.md, CLAUDE.md,
                        docs/adr/0001-..., docs/codebase/ARCHITECTURE.md,
                        docs/codebase/CONCERNS.md}
         quotes hidden_size or DEFAULT_EPOCHS at a value that does NOT
         equal the value in source code)

  RETURN collapse_or_overfit OR cpu_bound OR doc_drift
END FUNCTION
```

### Examples

Mọi ví dụ dưới lấy trực tiếp từ `train_log.txt` (run bắt đầu
`2026-05-15 18:32:01`, kết thúc `2026-05-15 18:38:35`, 200 epoch).

- **Epoch 1** — `train_loss = 44.39`, `eval_loss = 41.12`,
  `eval_exact_match = 0.000`, `eval_cer = 1.000`. Mong đợi: ≥ 1 ảnh đúng
  ở một epoch nào đó. Thực tế: 0/113 cả 200 epoch.
- **Epoch 10** — `train_loss = 3.5138` (≈ `log(25)`), `eval_loss = 3.515`.
  Mong đợi: model bắt đầu phân biệt class. Thực tế: stuck ở blank-collapse.
- **Epoch 18** — `eval_loss = 3.388` (đáy của run), `train_loss = 3.3867`.
  Mong đợi: đáy này tiếp tục giảm. Thực tế: từ đây eval_loss đi ngược lại.
- **Epoch 100** — `train_loss = 3.1573`, `eval_loss = 3.516`, gap = +0.36.
  Mong đợi: gap thu hẹp. Thực tế: gap tiếp tục mở.
- **Epoch 200 (cuối)** — `train_loss = 2.5765`, `eval_loss = 3.823`,
  gap = +1.25 nats, `eval_exact_match = 0.000`, `eval_cer = 0.949`,
  `[DONE] Best val_exact_match=0.0000 at epoch 0`. `eval_loss > log(25)`
  → val tệ hơn output uniform random.
- **GPU util** — `nvidia-smi` quan sát trung bình ~50% trong steady-state,
  12 CPU threads ở ~100%; throughput log: ~10.7 train-it/s, ~1.8s/epoch
  trên 20 batch (DataLoader bottleneck).
- **Doc drift** — `crnn_model.py` line 25 docstring: "BiLSTM: 2 layers,
  hidden=256"; `crnn_model.py` line 90 code: `hidden_size: int = 128`.
  `README.md`/`PIPELINE_SUMMARY.md`/`CLAUDE.md` quote `hidden=256` và
  `Epochs=50`. Run lỗi in `CRNN params: 2,186,553` chỉ khớp với
  `hidden=128`, không khớp với `hidden=256`.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors (3.1 – 3.10):**

- **3.1 Charset & class layout** — `CAPTCHA_CHARSET` byte-for-byte là
  `"ACDEFHJKLMNPQRTUVWXY3479"`, `CTC_BLANK_INDEX = 0`, `NUM_CLASSES = 25`,
  `CHAR_TO_IDX`/`IDX_TO_CHAR` không đổi mapping.
- **3.2 Input shape & normalize** — Mọi đường dẫn preprocess (cả
  `dataset_crnn._resize_and_normalize` và `inference_crnn._preprocess`)
  vẫn sản xuất tensor `(B, 3, 64, 320)`, ImageNet `mean=(0.485, 0.456,
  0.406)`, `std=(0.229, 0.224, 0.225)`.
- **3.3 ONNX contract** — `crnn_model.export_onnx` vẫn dùng opset 14,
  `input_names=["input"]`, `output_names=["logits"]`,
  `dynamic_axes={"input": {0: "batch"}, "logits": {1: "batch"}}`,
  `dummy = torch.randn(1, 3, 64, 320)`.
- **3.4 Resume flow** — `--resume` từ `captcha_crnn_last.pth` khôi phục
  `state_dict / optimizer / scheduler / scaler / epoch / best_val_em /
  best_epoch` và tiếp tục từ `epoch + 1`.
- **3.5 Windows compatibility** — Lệnh mặc định `python train_crnn.py`
  trên Windows + albumentations + CUDA 12.8 khởi động không lỗi pickle/
  spawn. `num_workers=0` vẫn là fallback hợp lệ; bất kỳ `num_workers > 0`
  được sử dụng phải pass smoke khởi động trên Windows.
- **3.6 Decode semantics** — `crnn_model.decode_greedy` và
  `decode_greedy_with_confidence` giữ ngữ nghĩa CTC greedy chuẩn (per-step
  argmax → collapse repeats → drop blanks). `inference_crnn._enforce_length`
  vẫn cắt nếu dài hơn 5, pad bằng `CAPTCHA_CHARSET[0]` ('A') nếu ngắn hơn.
- **3.7 Train/val split** — `dataset_crnn.create_crnn_datasets` vẫn dùng
  `val_split=0.15`, `seed=42`, val luôn từ real, dùng cùng `pd.sample`
  random_state để split deterministic.
- **3.8 Solver API & eval format** — `CRNNCaptchaSolver.{solve,
  solve_with_confidence, solve_batch, solve_batch_with_confidence}` giữ
  signature; `eval_crnn.evaluate` giữ output (exact_match, CER,
  per-position accuracy, top-10 confusions, low-confidence wrongs, verdict).
- **3.9 No synthetic data** — `python train_crnn.py` mặc định không gọi
  `generate_synthetic_crnn`, không đọc `data/synthetic_crnn/`, không đặt
  `use_synthetic=True`. Train mặc định chỉ dùng 754 ảnh real.
- **3.10 Hardware envelope** — VRAM peak ≤ 8GB và RAM peak ≤ 16GB ở
  `DEFAULT_BATCH_SIZE=32` trên Windows + i5-12400F + RTX 3060.

**Scope:**

Mọi cấu hình KHÔNG nằm trong tập `(augmentation_pipeline, base_lr,
warmup_steps, num_workers, persistent_workers, prefetch_factor,
hidden_size_in_docs, in-code comments mô tả hidden_size)` SHALL không
bị fix này chạm tới. Cụ thể: bất biến quan sát được qua các test
preservation dưới đây không được phép thay đổi giá trị/byte/format khi
chạy trước-vs-sau-fix.

**Note:** Behavior đúng kỳ vọng cho buggy inputs (Property 1) được định
nghĩa ở section "Correctness Properties" bên dưới. Section này chỉ liệt
kê những gì PHẢI **không** đổi.

## Hypothesized Root Cause

Phân tích từ `train_log.txt`, `dataset_crnn.py`, `train_crnn.py`,
`crnn_model.py` và biểu hiện CTC blank collapse trong literature cho thấy
nguyên nhân gốc là **tổ hợp 4 yếu tố cộng hưởng**, không phải một bug
đơn lẻ:

1. **Augmentation quá mạnh so với corpus 641 ảnh train** — `_build_albu_aug
   (strong=True)` áp Affine `rotate=(-12,12)`/`shear=(-5,5)`/
   `translate=(-0.06,0.06)`/`scale=(0.85,1.15)` ở `p=0.6`, kèm Perspective
   `scale=(0.02,0.08)` ở `p=0.3`, ColorJitter mạnh, GaussNoise, Blur,
   CoarseDropout. Mức distortion này tạo ra distribution train **rộng hơn
   nhiều** so với 113 ảnh val sạch → model học pattern augmentation chứ
   không học mapping image→text. Triệu chứng `train_loss ↓` trong khi
   `eval_loss ↑` từ epoch 18 trở đi là chữ ký kinh điển của augmentation
   overfit.

2. **CTC blank collapse do LR cao + warmup ngắn cho dataset nhỏ** —
   `DEFAULT_LR=1e-3` với AdamW + warmup chỉ `200` steps trên 20 step/epoch
   → warmup hoàn tất sau 10 epoch ở LR full rồi cosine decay rất chậm.
   Trên một corpus 641 ảnh với augmentation mạnh, gradient dồn về
   "predict-blank-everywhere" là cực tiểu cục bộ rất nông và hấp dẫn.
   Triệu chứng là `train_loss` mắc kẹt quanh `log(25) ≈ 3.219` từ epoch
   3 đến epoch 12 — bằng đúng entropy của uniform 25 classes.

3. **DataLoader CPU-bound trên Windows** — `num_workers = 0` hard-code
   với comment "tránh pickle issue với albumentations" nghĩa là toàn bộ
   pipeline albumentations chạy đồng bộ trên main thread. Trên i5-12400F,
   12 luồng CPU bão hòa làm augmentation cho từng batch trong khi GPU
   chờ → util ~50%. Đây là performance bug riêng nhưng cũng làm các chu
   kỳ thử nghiệm để debug correctness chậm hơn cần thiết.

4. **Doc/code drift làm khó chẩn đoán** — `crnn_model.py` line 25
   docstring quảng cáo "hidden=256" nhưng `__init__` đặt `hidden_size:
   int = 128`; README/PIPELINE_SUMMARY/CLAUDE/ADR-0001 đều viết
   `hidden=256` và `Epochs=50`. Bất kỳ ai đọc doc để tune sẽ baseline sai;
   con số `~2.18M params` quote trong doc chỉ đúng với code (`hidden=128`)
   chứ không đúng với doc (`hidden=256`).

Các nghi vấn được **bác bỏ** sau khi đọc code:
- *"Logits đảo trục T/B"* — không, `forward` permute đúng `(w, B, 512)`
  trước khi vào LSTM, FC ra `(T, B, num_classes)` đúng định dạng CTCLoss
  yêu cầu.
- *"Label mã hoá sai"* — không, `encode_label` dùng `CHAR_TO_IDX`
  1-based (`+1` offset cho blank=0) đúng convention CTC.
- *"`input_lengths` sai"* — không, `T_size = log_probs.size(0)` được fill
  vào `input_lengths` đúng `(B,)`.
- *"Bug ở scheduler bước trước optimizer"* — log có warning `Detected call
  of lr_scheduler.step() before optimizer.step()` ở batch đầu tiên do
  `scaler.step` skip một lần, nhưng đây là noise của AMP warm-up; LR vẫn
  ramp đúng từ epoch 1.

## Correctness Properties

Property 1: Bug Condition — Trained CRNN+CTC Generalises to Real
Validation Set And GPU is Saturated

_For any_ training run on `data/metadata.csv` (754 real images,
train=641/val=113, `seed=42`) trên hardware target (Windows + i5-12400F +
RTX 3060 8GB + CUDA 12.8) thoả `isBugCondition` ở trạng thái pre-fix, run
tương ứng sau khi áp fix SHALL đạt **đồng thời** cả 4 ngưỡng dưới đây:

1. `max over epochs of eval_exact_match ≥ 0.30` (≥ 34/113 ảnh decode đúng
   cả 5 ký tự ở ít nhất một checkpoint).
2. `min over epochs of eval_loss < log(NUM_CLASSES) = log(25) ≈ 3.219`
   (val loss xuống dưới ngưỡng "predict toàn blank") **và đồng thời**
   `< 3.388` (đáy của run lỗi tại epoch 18).
3. `min over epochs of (eval_loss − train_loss) ≤ 0.50` nats (chấm dứt
   mode phân kỳ `train_loss ↓ / eval_loss ↑`).
4. Mean GPU utilization quan sát qua `nvidia-smi
   --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000`
   trong cửa sổ liên tục 30s ở giữa training (loại trừ epoch đầu, eval
   step, ONNX export step) `≥ 80%` trên hardware target.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation — All Non-Training-Tuning Behavior Identical

_For any_ behavior X thuộc tập preservation `{charset & class layout,
input shape & normalize, ONNX export contract, --resume flow, Windows
boot smoke, decode semantics, _enforce_length, train/val split,
CRNNCaptchaSolver API, eval_crnn output format, "no synthetic" default,
hardware envelope}`, output của X ở bản fix SHALL **bằng** output của X ở
bản gốc trên cùng input — bytewise cho artefacts deterministic (charset
strings, file headers, ONNX I/O contract), shape+dtype+API-signature cho
artefacts có yếu tố ngẫu nhiên (split với cùng seed, decode trên cùng
logits).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

Property 3: Doc/Code Reconciliation — Single Source of Truth

_For any_ document trong tập `{README.md, PIPELINE_SUMMARY.md, CLAUDE.md,
docs/adr/0001-crnn-ctc-over-softmax.md, docs/codebase/ARCHITECTURE.md,
docs/codebase/CONCERNS.md}` nhắc đến `hidden_size`, `DEFAULT_EPOCHS`,
`DEFAULT_BATCH_SIZE`, hoặc `count_parameters()`, doc đó SHALL hoặc tham
chiếu hằng số theo tên (e.g. "see `crnn_model.CRNN.__init__` for canonical
`hidden_size`"), hoặc quote đúng giá trị canonical đã định nghĩa trong
code (`hidden_size=128`, `DEFAULT_EPOCHS=200`, `DEFAULT_BATCH_SIZE=32`,
`count_parameters() == 2_186_553`).

**Validates: Requirements 2.5**

## Fix Implementation

### Changes Required

Giả định root cause analysis đúng (sẽ được verify bằng test exploratory
trước khi áp fix), các thay đổi cụ thể:

#### A. `dataset_crnn.py` — Augmentation tuning

**Function**: `_build_albu_aug(strong: bool = True)` và `_TRAIN_AUG`/
`_TV_TRAIN_AUG` ở module-level.

**Specific Changes**:
1. **Hạ độ mạnh Affine**:
   - `rotate=(-12, 12)` → `rotate=(-4, 4)`
   - `shear=(-5, 5)` → `shear=(-2, 2)`
   - `translate_percent=(-0.06, 0.06)` → `(-0.03, 0.03)`
   - `scale=(0.85, 1.15)` → `(0.92, 1.08)`
   - `p=0.6` → `p=0.5`
2. **Hạ Perspective**: `scale=(0.02, 0.08)` → `(0.01, 0.04)`, `p=0.3` →
   `p=0.2`.
3. **Tone down ColorJitter** (`RandomBrightnessContrast` +
   `HueSaturationValue`):
   - `brightness_limit=0.2` → `0.15`, `contrast_limit=0.2` → `0.15`
   - `hue_shift_limit=10` → `5`, `sat_shift_limit=20` → `12`,
     `val_shift_limit=15` → `10`
4. **Hạ Noise + Blur**:
   - `GaussNoise var_limit=(5.0, 25.0)` → `(3.0, 12.0)`, `p=0.4` → `p=0.3`
   - `OneOf(GaussianBlur, MotionBlur) p=0.25` → `p=0.15`
5. **Hạ CoarseDropout** (giữ vì giúp robust với noise nền nhưng giảm
   diện tích):
   - `max_holes=4` → `2`, `max_height=8` → `5`, `max_width=8` → `5`
   - `p=0.2` → `p=0.1`
6. **Đồng bộ torchvision fallback `_TV_TRAIN_AUG`** với cùng hệ số (rotate
   `±4`, translate `0.03`, scale `(0.92, 1.08)`, shear `2`, brightness/
   contrast/saturation `0.15`, hue `0.04`) để hai nhánh có cùng độ mạnh.
7. **Pickleable cho multi-worker DataLoader**: `_TRAIN_AUG` ở module-level
   đã pickle được (albumentations 1.4.3 hỗ trợ); xác nhận bằng smoke test
   với `num_workers=4`. Nếu phát sinh issue, gói augmentation vào hàm
   top-level `def _apply_train_aug(rgb_array)` trả ảnh đã augment, được
   tham chiếu từ `Dataset.__getitem__`.

**Lưu ý preservation**: `_resize_and_normalize`, `_MEAN`, `_STD`,
`CRNNCaptchaDataset.__init__` (text strip/upper, len==5 filter),
`collate_fn` shape contract, `create_crnn_datasets` split logic
(`val_split=0.15`, `seed=42`, val từ real) **không** đổi.

#### B. `train_crnn.py` — Hyperparam + DataLoader workers

**Function**: `main` (block "Data" và đoạn "Optimizer/Scheduler"),
constants ở top of file.

**Specific Changes**:
1. **Hạ base LR**: `DEFAULT_LR: float = 1e-3` → `5e-4`. Phù hợp với corpus
   ~641 ảnh + AdamW + warmup-cosine; giảm rủi ro CTC collapse.
2. **Mở rộng warmup tương đối, giữ floor tuyệt đối**: thay
   `warmup_steps=min(WARMUP_STEPS, total_steps // 10)` bằng
   `warmup_steps = max(WARMUP_STEPS, steps_per_epoch * 2)` — đảm bảo
   warmup ≥ 2 epoch để không "đốt" qua warmup chỉ trong 10 epoch trên
   dataset nhỏ. Giữ `WARMUP_STEPS = 200` làm floor.
3. **Bật multi-worker DataLoader (Windows-safe)**:
   - Thêm CLI flag `--num-workers` (default `None` → auto).
   - Auto policy: nếu `--num-workers` không truyền và `os.name == "nt"`,
     đặt `num_workers = min(4, os.cpu_count() // 2)` nhưng `0` nếu
     albumentations import fail (fallback về torchvision).
   - Trên Linux/macOS giữ logic auto = `min(8, os.cpu_count() // 2)`.
   - Đặt `persistent_workers=True` khi `num_workers > 0`.
   - Đặt `prefetch_factor=4` khi `num_workers > 0`.
   - `pin_memory=use_amp` (giữ nguyên).
   - **Quan trọng cho Windows**: bọc lệnh khởi tạo DataLoader trong
     `if __name__ == "__main__":` (đã có trong `train_crnn.py`); xác
     nhận `_TRAIN_AUG` có thể pickle dưới spawn start method.
4. **Cập nhật comment**: thay comment "`num_workers = 0` # Windows: 0 để
   tránh pickle issue với albumentations" bằng comment giải thích auto-
   policy mới và rằng albumentations 1.4.3 + Compose ở top-level
   pickleable trên Windows spawn.
5. **Thêm log dòng `num_workers` thực tế** vào `logger.info` để dễ debug
   regression sau này.

**Lưu ý preservation**: `CHECKPOINT_PATH`, `LAST_CHECKPOINT_PATH`,
`ONNX_PATH`, `DEFAULT_EPOCHS=200`, `DEFAULT_BATCH_SIZE=32`,
`GRAD_CLIP_NORM=5.0`, structure của resume payload (keys
`state_dict/optimizer/scheduler/scaler/epoch/best_val_em/best_epoch`),
ONNX export ở cuối, eval log dict format **không** đổi.

#### C. `crnn_model.py` — Architecture canonicalization

**Function**: docstring đầu file + `CRNN.__init__`.

**Specific Changes**:
1. **Xác định canonical**: chốt `hidden_size: int = 128` (giữ default
   hiện tại — đây là giá trị thực tế đang chạy và in `2,186,553` params
   trong `train_log.txt`). FC layer `Linear(2 * hidden_size, num_classes)
   = Linear(256, 25)` — không đổi code, chỉ xác nhận tính nhất quán.
2. **Sửa docstring nội bộ** đang viết "BiLSTM: 2 layers, hidden=256,
   dropout 0.2 → (T=80, B, 512) (bidir = 2*hidden)" thành "BiLSTM: 2
   layers, hidden=128, dropout 0.2 → (T=80, B, 256) (bidir = 2*hidden)";
   sửa kèm dòng tổng kết "(B, 512, h≈4, w=80)" và "(T=80, B, 512)" cho
   trùng với hidden=128.
3. **Thêm note tham chiếu**: thêm dòng `# Canonical params:
   count_parameters() == 2_186_553 with hidden_size=128. Update
   PIPELINE_SUMMARY.md / README.md / CLAUDE.md if you change this.` ngay
   trước `class CRNN`.
4. **Không đổi**: `CAPTCHA_CHARSET`, `NUM_CLASSES`, `CTC_BLANK_INDEX`,
   `CHAR_TO_IDX`, `IDX_TO_CHAR`, `INPUT_HEIGHT=64`, `INPUT_WIDTH=320`,
   forward/encode/decode/save/load/export_onnx logic.

#### D. Doc/code reconciliation

**Files**: `README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`,
`docs/adr/0001-crnn-ctc-over-softmax.md`,
`docs/codebase/ARCHITECTURE.md`, `docs/codebase/CONCERNS.md`.

**Specific Changes**:
1. **Hyperparam tables** trong `README.md`/`PIPELINE_SUMMARY.md`/
   `CLAUDE.md`: thay `BiLSTM hidden=256` → `BiLSTM hidden=128 (see
   crnn_model.CRNN default)`, `Epochs=50` → `Epochs=200 (see
   train_crnn.DEFAULT_EPOCHS)`. Quote `count_parameters() == 2,186,553`.
2. **ADR-0001** — thêm 1 đoạn "Implementation note (2026-05-15): canonical
   `hidden_size=128`, không phải 256 như draft ban đầu; param count
   2,186,553 verified bằng `CRNN().count_parameters()`."
3. **`docs/codebase/ARCHITECTURE.md`** mục "Known Architectural Risks":
   gạch đầu dòng "Doc/code drift on hyperparameters" sửa thành "Resolved
   (2026-05-15): canonical hidden_size=128, DEFAULT_EPOCHS=200; doc đã
   sync."
4. **`docs/codebase/CONCERNS.md`** mục "Top Risks" entry "Documentation/
   code drift on the BiLSTM hidden size and default epochs": status →
   "Resolved" với reference đến commit fix.
5. **Không** chạm `CONTEXT.md`, `docs/research_strategy_*`, hoặc bất kỳ
   doc nào ngoài 6 file trên.

## Testing Strategy

### Validation Approach

Hai pha. Pha 1 chạy trên code **chưa fix** để **xác nhận** root cause
(test exploratory phải fail). Pha 2 chạy trên code **đã fix** để verify
fix đạt cả 3 properties (Bug-condition correctness, Preservation,
Doc/code reconciliation).

Thử nghiệm correctness chính (Property 1) có chi phí cao (mỗi training
run ≈ 6 phút trên RTX 3060 cho 200 epoch theo log; smoke 20 epoch ≈ 1
phút). Vì vậy chia nhỏ thành smoke (20 epoch) + full (200 epoch) như
mô tả ở mục Integration Tests.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples chứng minh `isBugCondition` đang holds
trên code chưa fix, AND xác nhận giả thuyết root cause (augmentation +
LR + workers + doc-drift) trước khi áp fix.

**Test Plan**: Một phần đã có sẵn dưới dạng `train_log.txt` của run lỗi.
Bổ sung thêm các kiểm tra ngắn để confirm/refute từng nhánh hypothesis.
Nếu một nhánh refute, cần re-hypothesize trước khi áp phần fix tương
ứng.

**Test Cases** (tất cả chạy trên code unfixed):
1. **Replay lỗi từ log có sẵn** — Đọc `train_log.txt` qua một script
   parse `eval_exact_match`, `eval_loss`, `train_loss`, assert
   `max(eval_exact_match) == 0.000` AND `min(eval_loss) >= 3.388` AND
   final `eval_loss > log(25)`. PASS = bug confirmed (will pass on
   unfixed log).
2. **Augmentation visualization** — Sample 16 ảnh từ `train_ds.augment=
   True` qua `_TRAIN_AUG`, render side-by-side với ảnh gốc; đo
   `pixel-wise std` và `histogram divergence` augmented vs original.
   Expected: divergence cao đến mức một con người cũng đọc lệch ký tự,
   chứng minh augmentation là nguồn gap distribution.
3. **LR floor experiment** — Train smoke 20 epoch với `--lr 5e-4`
   nhưng giữ nguyên augmentation hiện tại; observe train_loss/eval_loss.
   Nếu eval_loss vẫn ≥ 3.388 → LR không phải mode chính → hypothesize
   thêm augmentation. Nếu eval_loss < 3.219 → LR là mode chính, có thể
   chỉ cần fix B + C + D không cần A.
4. **Augmentation OFF experiment** — Train smoke 20 epoch với
   `--no-augment`, giữ nguyên LR=1e-3 và num_workers=0; observe. Nếu
   eval_exact_match > 0.10 → augmentation chính là yếu tố overfit, fix
   A bắt buộc. Nếu vẫn 0.000 → cần đào sâu hơn (CTC config, mã hoá
   label).
5. **GPU util sampling unfixed** — Chạy training 30s sau warmup, dùng
   `nvidia-smi -lms 1000` lấy mẫu 30 lần, assert `mean < 80%`. PASS =
   CPU-bound confirmed (will pass on unfixed code).
6. **Doc drift grep** — `grep -E "hidden(_size)?\s*=?\s*256|Epochs\s*=
   \s*50"` qua 6 file doc liệt kê ở Property 3, assert có ≥ 1 match.
   PASS = drift confirmed.

**Expected Counterexamples**:
- (1), (5), (6) PASS trên unfixed → confirm bug.
- (3) sẽ chỉ giảm loss đôi chút, không vượt threshold (confirm A là cần
  thiết).
- (4) sẽ cho `eval_exact_match` tăng đáng kể (>= 0.10) (confirm
  augmentation là driver chính).
- Possible alternative causes: nếu (4) vẫn cho 0, các nghi vấn đã bác bỏ
  ở section Hypothesized Root Cause cần re-test (logits permute,
  encode_label off-by-one, input_lengths sai).

### Fix Checking

**Goal**: Verify với mọi input thoả `isBugCondition`, `train_crnn.main`
sau fix produces output thoả Property 1 (đồng thời cả 4 ngưỡng) và
Property 3 (doc/code reconciled).

**Pseudocode:**

```
FOR ALL run WHERE isBugCondition(run.config_pre_fix) DO
  // Apply fix
  apply_fix_A_dataset_crnn()
  apply_fix_B_train_crnn()
  apply_fix_C_crnn_model()
  apply_fix_D_docs()

  // Re-run with same dataset/seed/hardware
  metrics := train_crnn_main(run.config_post_fix)
  gpu_util := measure_gpu_steady_state(window_seconds=30)

  ASSERT max(metrics.eval_exact_match) >= 0.30
  ASSERT min(metrics.eval_loss)        <  3.219
  ASSERT min(metrics.eval_loss)        <  3.388
  ASSERT min(metrics.eval_loss - metrics.train_loss) <= 0.50
  ASSERT mean(gpu_util)                >= 0.80
  ASSERT doc_grep_for_stale_values()   == empty
END FOR
```

### Preservation Checking

**Goal**: Verify với mọi input không liên quan tới hyperparam/augmentation/
workers/doc, output bản fix == output bản gốc.

**Pseudocode:**

```
FOR ALL behavior B in PreservationSet DO
  output_pre  := run_behavior_B_on_unfixed_code()
  output_post := run_behavior_B_on_fixed_code()
  ASSERT output_pre == output_post  (modulo declared randomness)
END FOR
```

**Testing Approach**: Property-based testing được dùng cho Preservation
check vì:
- Mỗi behavior có không gian input lớn (mọi label hợp lệ, mọi ảnh hợp
  lệ, mọi seed) — PBT generate samples ngẫu nhiên ổn định hơn unit test
  thủ công.
- Các invariant như "encode/decode round-trip cho mọi label hợp lệ", "no
  synthetic data path lazy-loaded khi default flag", "split với
  seed=42 deterministic", "tensor shape (B,3,64,320)" tự nhiên là
  property-based.
- Dùng `hypothesis` (Python PBT framework) — nhẹ, không thêm dep production
  (chỉ dev-time).

**Test Plan**: Đọc behavior trên unfixed code trước (capture output làm
oracle), rồi viết PBT trên fixed code so sánh.

**Test Cases**:
1. **Charset preservation (3.1)** — Property: `len(CAPTCHA_CHARSET) ==
   24 ∧ NUM_CLASSES == 25 ∧ CTC_BLANK_INDEX == 0 ∧ CAPTCHA_CHARSET ==
   "ACDEFHJKLMNPQRTUVWXY3479"`. Static, single assertion.
2. **Input shape preservation (3.2)** — Property: `for all valid input
   image (HxWx3 uint8), _resize_and_normalize(img).shape == (3, 64, 320)
   ∧ dtype == float32`. PBT với `hypothesis.strategies` sinh ngẫu nhiên
   H, W ∈ [16, 1024], C=3.
3. **ONNX contract (3.3)** — Sau khi train smoke 1 epoch, chạy
   `export_onnx`, load ONNX qua `onnx.load`, assert input name = "input",
   output name = "logits", input shape `[batch, 3, 64, 320]`, opset_version
   == 14.
4. **Resume flow (3.4)** — Smoke train 2 epoch → save → resume với
   `--resume` → train thêm 2 epoch. Property: `final_epoch == 4 ∧
   restored_optimizer.param_groups[0]['lr'] != fresh_optimizer ∧
   best_val_em ∈ resumed_state ∧ best_epoch ∈ resumed_state`.
5. **Windows boot smoke (3.5)** — Trên Windows, chạy `python
   train_crnn.py --epochs 1` với cấu hình post-fix (kể cả
   `num_workers > 0`); assert process exit code 0 và không có
   `BrokenPipeError`/`PicklingError` trong stderr. Cũng test
   `python train_crnn.py --num-workers 0 --epochs 1` để confirm fallback.
6. **Decode semantics (3.6)** — Property: `for all valid 5-char text T,
   decode_greedy(simulate_logits_from_text(T)) == [T]`; cũng property:
   `for all model output logits L, _enforce_length(decode_greedy(L)[0])
   has length == 5 ∧ if len(decoded) < 5 then padded char ==
   CAPTCHA_CHARSET[0] == 'A'`. Hypothesis sinh `T` ngẫu nhiên từ charset.
7. **Train/val split determinism (3.7)** — Property: `for two calls of
   create_crnn_datasets(seed=42), train_ds.df["filename"].tolist() ==
   train_ds_2.df["filename"].tolist() ∧ val_ds.df["filename"].tolist()
   == val_ds_2.df["filename"].tolist()`. Cũng assert `len(val_ds) /
   (len(train_ds) + len(val_ds)) ≈ 0.15` trên cùng metadata.csv.
8. **Solver API (3.8)** — Property: với checkpoint giả (CRNN khởi tạo
   ngẫu nhiên, `save_crnn`), `CRNNCaptchaSolver.solve(path)` trả
   `str` length=5 cho mọi ảnh hợp lệ; `solve_with_confidence` trả
   `(str, float)` với float ∈ [0, 1]; `solve_batch(paths)` length ==
   `len(paths)`; `solve_batch_with_confidence` cùng cardinality.
9. **eval_crnn output format (3.8)** — Smoke run `eval_crnn.evaluate`
   trên 10 ảnh; assert format output có các dòng "Exact match", "CER",
   "Per-position accuracy", "Top-10 confusions", "Verdict".
10. **No synthetic data default (3.9)** — Static analysis: `grep -n
    "synthetic_crnn\|generate_synthetic_crnn" train_crnn.py` chỉ xuất
    hiện trong CLI `--use-synthetic` flag và `if use_synthetic:` branch.
    Smoke runtime: chạy `python train_crnn.py --epochs 1` với
    `data/synthetic_crnn` đã rename; assert không có FileNotFoundError.
11. **Hardware envelope (3.10)** — Smoke `python train_crnn.py
    --epochs 1` trên RTX 3060: query `torch.cuda.max_memory_allocated()`
    cuối epoch, assert `< 8 * 1024**3`; đo `psutil.Process().memory_info(
    ).rss` peak, assert `< 16 * 1024**3`.

### Unit Tests

- `test_charset_invariants.py` — assert `CAPTCHA_CHARSET`, `NUM_CLASSES`,
  `CTC_BLANK_INDEX`, mappings 1-based đúng (preservation 3.1).
- `test_encode_decode_roundtrip.py` — round-trip `encode_label →
  simulated_logits → decode_greedy` cho mọi 5-char text hợp lệ
  (preservation 3.6).
- `test_enforce_length.py` — boundary cases len 0, 1, 4, 5, 6, 10
  (preservation 3.6).
- `test_resize_and_normalize.py` — đa kích thước input → output đúng
  `(3, 64, 320)` float32, mean/std stats sau normalize ≈ ImageNet
  expected (preservation 3.2).
- `test_split_determinism.py` — hai lần `create_crnn_datasets(seed=42)`
  cho cùng filenames ở mỗi split (preservation 3.7).
- `test_canonical_constants.py` — assert `CRNN().count_parameters() ==
  2_186_553`, `CRNN().rnn.hidden_size == 128`, `train_crnn.DEFAULT_EPOCHS
  == 200`, `train_crnn.DEFAULT_BATCH_SIZE == 32`, `train_crnn.DEFAULT_LR
  == 5e-4` (correctness 2.5 + post-fix LR).
- `test_no_synthetic_default.py` — static AST check + runtime smoke
  (preservation 3.9).
- `test_doc_consistency.py` — grep 6 file doc cho các giá trị stale
  (`hidden=256`, `Epochs=50`, `~2.18M params` trong doc nói `hidden=256`)
  → assert empty (Property 3, validates 2.5).

### Property-Based Tests

Dùng `hypothesis` cho test space lớn:

- `prop_encode_decode_roundtrip` — `@given(st.text(alphabet=
  CAPTCHA_CHARSET, min_size=5, max_size=5))` → encode → simulate
  perfectly-confident logits → decode_greedy → assert recovered == input.
  Validates Property 2 (decode semantics).
- `prop_enforce_length_invariant` — `@given(st.text(alphabet=
  CAPTCHA_CHARSET + "Z", max_size=20))` → `_enforce_length(s)` length
  exactly 5 cho mọi input. (Z là char ngoài charset để robust).
- `prop_resize_shape` — `@given(st.integers(16, 1024),
  st.integers(16, 1024))` → tạo H×W×3 uint8 → `_resize_and_normalize`
  → assert shape `(3, 64, 320)`, dtype float32.
- `prop_split_seed_determinism` — `@given(st.integers(0, 1000))` cho
  seed → 2 lần `create_crnn_datasets` cho cùng output filename lists.
- `prop_collate_fn_shapes` — `@given(st.lists(<sample tuples>, min_size=
  1, max_size=8))` → `collate_fn(batch)` trả dict đúng shape, đúng
  `sum(label_lengths) == len(labels)`.

### Integration Tests

Có chi phí cao hơn (chạy GPU thật), dùng làm gate cuối:

1. **Smoke training (correctness sub-target)** — `python train_crnn.py
   --epochs 20` post-fix: assert `max(eval_exact_match) ≥ 0.05` (one-
   percent floor để thoát collapse) AND `min(eval_loss) < 3.0` AND gap
   ≤ 0.7. Đây là smoke nhanh trước khi commit chạy 200-epoch full.
2. **Full training (correctness Property 1)** — `python train_crnn.py
   --epochs 200` post-fix trên hardware target: assert tất cả 4 ngưỡng
   của Property 1 hold.
3. **GPU util steady-state** — Trong khi (1) đang chạy, trigger script
   `monitor_gpu_util.py` lấy mẫu `nvidia-smi -lms 1000` trong 30s ở
   epoch giữa (vd epoch 5–6); assert mean ≥ 0.80.
4. **Resume integration** — `python train_crnn.py --epochs 4` →
   `python train_crnn.py --epochs 8 --resume`: assert epoch counter
   tiếp tục từ 5, best_val_em được carry forward.
5. **End-to-end eval contract** — Sau full training, chạy `python
   eval_crnn.py`; assert format output ổn định và verdict line khớp
   với `eval_exact_match` đo được.
6. **Windows multi-worker boot** — Trên Windows, `python train_crnn.py
   --epochs 1 --num-workers 4` phải boot không lỗi pickle/spawn (test
   3.5 mở rộng).
7. **ONNX export contract** — Sau full training, `python -c "import
   onnx; m = onnx.load('captcha_crnn_model.onnx'); ..."`: assert input
   name "input", output name "logits", input shape `[batch_dim, 3, 64,
   320]`, opset 14.
