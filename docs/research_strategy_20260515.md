# Nghiên cứu chiến lược đạt 90%+ exact_match (UPDATED)

| Field | Value |
|---|---|
| **Date** | 2026-05-15 (updated after 200 epochs trên GPU) |
| **Subject** | Phân tích data + training log + đề xuất strategy đạt target |
| **Current state** | 0% exact_match sau 200 epochs (CTC complete failure) |
| **Target** | exact_match ≥ 90%, CER ≤ 10% |

---

## 1. Phân tích training log mới (200 epochs trên GPU)

### 1.1 Hardware

Train log mới chạy trên **GPU** (eval_samples_per_second = 800-880, ~0.14s/eval batch). Khẳng định bởi:
- Eval runtime: 0.13-0.17s (CPU sẽ là 3-5s)
- Speed: 27-31 steps/second

### 1.2 Toàn bộ training (200 epochs)

| Epoch | train_loss | eval_loss | eval_cer | eval_em | LR |
|---|---|---|---|---|---|
| 1 | 44.39 | 41.12 | 1.000 | 0.000 | 1e-04 (warmup) |
| 2 | 29.64 | 9.28 | 1.000 | 0.000 | 2e-04 |
| 3 | 4.56 | 3.63 | 1.000 | 0.000 | 3e-04 |
| 4 | 3.54 | 3.52 | 1.000 | 0.000 | 4e-04 |
| 10 | 3.51 | 3.52 | 1.000 | 0.000 | 1e-03 |
| 50 | ~3.30 | ~3.36 | 0.94 | 0.000 | ~9e-04 |
| 100 | ~2.63 | ~3.20 | 0.88 | 0.000 | ~5e-04 |
| 145 | 2.03 | 3.29 | 0.844 | 0.000 | 2e-04 |
| 171 | 2.62 | 3.79 | 0.935 | 0.000 | 6.6e-05 |
| 200 | 2.58 | 3.82 | 0.949 | 0.000 | 1e-05 |

### 1.3 Quan sát quan trọng

**Train loss vs eval loss:**
- Train loss: 44 → 3.5 (epoch 4) → 2.58 (epoch 200) — giảm tốt
- Eval loss: 41 → 3.5 (epoch 4) → **3.82 (epoch 200)** — TĂNG ngược lại

**→ Đây là OVERFITTING NẶNG.** Model học train data nhưng không generalize.

**Best CER:** 0.919 ở epoch 31-33 (đầu training, trước khi overfit). Sau đó CER tăng dần lên 0.949.

**Best exact_match:** 0.000 ở MỌI epoch — model chưa từng predict đúng 1 ảnh nào.

### 1.4 Phân tích CTC behavior

Train_loss = 2.58 với CTC blank token = 0:
- Random init: ~3.22 (log(25) = 3.22)
- Train_loss < 3.22 nghĩa là model đã học được patterns
- Nhưng eval_loss = 3.82 > random → model output **tệ hơn random** trên val

**Hypothesis:** Model memorize training augmentation patterns, không học được captcha decode.

---

## 2. Root cause analysis (cập nhật)

### 2.1 Vấn đề chính: Augmentation phá shape ký tự

Pipeline hiện tại (`dataset_crnn.py`):
```python
A.Affine(rotate=(-12,12), translate_percent=(-0.06,0.06), scale=(0.85,1.15), shear=(-5,5), p=0.6)
A.Perspective(scale=(0.02,0.08), p=0.3)
```

**Tác động:**
- Rotate ±12° + shear ±5° → ký tự `K` có thể trông giống `X`, `Y`, `4`
- Perspective 0.02-0.08 → bẻ cong text strong
- Áp dụng với p=0.6 → 60% batches bị augment mạnh

**Kết quả:** Train data sau augment có thể không match label (`A` augment thành `J`-like nhưng label vẫn `A`) → model học sai mapping.

### 2.2 Vấn đề thứ hai: Data quá ít

641 train samples × 38 batches/epoch (batch=16) = ~24K iterations/epoch
Sau 200 epochs = 4.8M iterations

Tuy nhiên, mỗi sample chỉ có ~5 ký tự → tổng 5*641 = 3,205 char examples
Với 24 classes, mỗi class chỉ có ~134 examples. **Quá ít cho CTC**.

CTC cần align T=79 timesteps với 5 chars → 15 timesteps/char. Model phải học ALL alignment paths cho mỗi char trong mỗi sample. Chỉ 134 examples/class không đủ.

### 2.3 Vấn đề thứ ba: Model size

Đã giảm từ 8.7M xuống 2.18M. Với augmentation mạnh + data ít:
- Quá nhỏ → underfit
- Combined với augmentation → train loss vẫn cao (2.58)

### 2.4 Vấn đề thứ tư: Học rate quá cao đầu

Warmup 200 steps → LR đạt 1e-3 ở epoch 10. Tại đó:
- Train_loss đã collapse về 3.51 (chỉ predict blank)
- Khi LR cao + data ít → model lock vào blank prediction
- Cosine decay xuống không thoát được khỏi local minimum này

---

## 3. Strategy update — 5 giải pháp ưu tiên
[
DONT DO THAT
### #1 (CRITICAL): Generate synthetic data 50K+

```bash
python generate_synthetic_crnn.py --count 50000
```

Lý do: 641 samples không đủ. 50K synthetic + 641 real → ~50K diverse examples.
Synthetic renderer đã calibrated từ 754 real (BG, font, overlap, rotation).

**Expected impact:** +30-50% accuracy
]
### #2 (CRITICAL): Giảm augmentation

```python
# dataset_crnn.py — replace strong aug
A.Affine(rotate=(-3, 3), translate_percent=(-0.02, 0.02), scale=(0.98, 1.02), p=0.3)
# REMOVE: shear, perspective
A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3)
A.GaussNoise(var_limit=(3.0, 10.0), p=0.2)
# REMOVE: HueSaturationValue, blur, CoarseDropout
```

**Expected impact:** Tránh CTC collapse, +10-20% accuracy

### #3 (HIGH): Restore model 8.7M params

```python
# crnn_model.py
class CRNN:
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256):  # was 128
        # CNN: 32 → 64 → 128 → 256 → 512 (was: 32 → 64 → 128 → 256)
```

**Expected impact:** +5-10% từ tăng capacity

### #4 (HIGH): Curriculum learning — train synthetic trước, real sau

```python
# Phase A: Train chỉ synthetic 30 epochs (dễ, ổn định)
# Phase B: Fine-tune trên synthetic + real 50 epochs
```

**Expected impact:** Tránh CTC collapse ngay từ đầu, +10% stability

### #5 (MEDIUM): Giảm LR đầu

```python
WARMUP_STEPS: int = 1000  # was 200
DEFAULT_LR: float = 5e-4  # was 1e-3
```

**Expected impact:** Tránh collapse blank, +5%

---

## 4. Pipeline khuyến nghị

```
Bước 1: Sửa code (3 file)
  - crnn_model.py: hidden=256, channels lớn hơn
  - dataset_crnn.py: augment NHẸ
  - train_crnn.py: lr=5e-4, warmup=1000, hỗ trợ 2-phase

BỎ QUA Bước 2: Generate 50K synthetic
  python generate_synthetic_crnn.py --count 50000

Bước 3: Train Phase A (synthetic only)
  python train_crnn.py --use-synthetic --no-real --epochs 30

Bước 4: Train Phase B (real)
  python train_crnn.py --use-synthetic --epochs 50 --resume

Bước 5: Eval
  python eval_crnn.py
  → kỳ vọng 90%+

Bước 6: Self-training (nếu < 90%)
  python self_train.py
  python eval_crnn.py
  → kỳ vọng 85-92%
```

---

## 5. Agent Skills phù hợp

### 5.1 Skills đã cài

| Skill | Mục đích | Trạng thái |
|---|---|---|
| `acquire-codebase-knowledge` | Phân tích codebase | Đã cài |

### 5.2 Đánh giá Skills cho ML/CAPTCHA

**Tìm trên các nguồn:**
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) — 4.6K★, 200+ skills
- [agentskills.io](https://agentskills.io)
- [github/awesome-copilot](https://github.com/github/awesome-copilot)
- [skillmatic-ai/awesome-agent-skills](https://github.com/skillmatic-ai/awesome-agent-skills)

**Kết luận:** KHÔNG có Agent Skill nào chuyên cho:
- PyTorch training pipeline
- CRNN/CTC debugging
- CAPTCHA OCR

Skills hiện có chủ yếu cho:
- Code review (`pr-review`)
- Documentation generation
- Git workflow
- Deployment automation
- Codebase analysis (như skill đã cài)

→ **Agent Skills KHÔNG giúp giải quyết vấn đề ML hiện tại.**

### 5.3 Reference repos (code, không phải skills)

| Repo | Stars | Phù hợp | Lý do |
|---|---|---|---|
| [sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer) | 1K | ⭐⭐⭐⭐⭐ | CRNN+CTC pipeline hoàn chỉnh, đã handle CTC collapse |
| [GabrielDornelles/pytorch-ocr](https://github.com/GabrielDornelles/pytorch-ocr) | ~ | ⭐⭐⭐⭐ | Framework CRNN+CTC+Attention |
| [hivaze/Captcha-OCR-Models](https://github.com/hivaze/Captcha-OCR-Models) | ~ | ⭐⭐⭐⭐ | So sánh nhiều OCR architectures cho captcha |
| [namdvt/CAPTCHA-Recognition-using-CRNN](https://github.com/namdvt/CAPTCHA-Recognition-using-CRNN) | ~ | ⭐⭐⭐ | CRNN+CTC PyTorch implementation |
| [YenLinWu/CRNN_with_CTC_Loss](https://github.com/YenLinWu/CRNN_with_CTC_Loss) | 8 | ⭐⭐⭐ | CRNN cho captcha 5-char fixed |

---

## 6. Diagnostic checklist trước khi train lại

- [ ] Synthetic data đã sinh chưa? (`data/synthetic_crnn/metadata.csv` exists?)
- [ ] Augmentation đã giảm? (rotate ≤ 5°, no shear/perspective)
- [ ] Model size 8.7M? (`hidden_size=256`, channels max=512)
- [ ] Warmup steps ≥ 1000? (tránh LR cao quá sớm)
- [ ] Decode greedy có drop blank đúng? (verify với `decode_greedy_with_confidence`)
- [ ] Sample 5 ảnh real → predict thử → có ra chuỗi không hay rỗng?

---

## 7. Time estimates (RTX 3060)

| Task | Thời gian |
|---|---|
| Generate 50K synthetic | 15-20 phút |
| Phase A (synthetic only, 30 epochs) | 25-35 phút |
| Phase B (synthetic + real, 50 epochs) | 60-80 phút |
| Eval | 30 giây |
| Self-train round 2 | 10-15 phút |
| **Tổng** | **~2-3 giờ** |

---

## 8. Confidence Assessment

| Claim | Confidence | Evidence |
|---|---|---|
| Augmentation quá mạnh là root cause | **High** | Train_loss giảm tốt nhưng eval_loss tăng (overfitting trên augment patterns) |
| 641 samples không đủ | **High** | 134 examples/class quá ít cho CRNN+CTC |
| Synthetic data → 70-85% | **High** | Research doc + ADR-0002 đã thiết kế cho strategy này |
| Self-training → 90%+ | **Medium** | Phụ thuộc Phase 1 đạt ≥ 70% trước |
| Agent Skills KHÔNG giúp ML | **High** | Skills hiện tại chỉ cho code workflow |

---

## 9. Sources

**Empirical data:**
- `train_log.txt` (3934 lines, 200 epochs, GPU run)
- `data/metadata.csv` (754 labels, char distribution analyzed)

**Project docs:**
- `research_minecraft_map_captcha_20260515.md`
- `docs/adr/0001-crnn-ctc-over-softmax.md`
- `docs/adr/0002-synthetic-first-training.md`

**External:**
- [agentskills.io](https://agentskills.io) — Agent Skills spec
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) — 200+ skills directory
- [sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer) — CRNN+CTC reference
- Graves et al. "Connectionist Temporal Classification" — CTC paper

---

## 10. Verdict

**Pipeline hiện tại có 4 vấn đề kỹ thuật rõ ràng:**

1. Augmentation quá mạnh (rotate ±12°, shear, perspective) → CTC overfit
2. Thiếu synthetic data (chỉ 641 samples)
3. Model bị giảm size (2.18M, đáng lẽ 8.7M)
4. LR quá cao quá sớm (1e-3 ở epoch 10)

**Các vấn đề này KHÔNG sửa được bằng train lâu hơn.** Sau 200 epochs vẫn 0% exact_match → cần sửa code.

**Action items theo thứ tự:**
1. Sửa 3 file (`crnn_model.py`, `dataset_crnn.py`, `train_crnn.py`)
2. Sinh synthetic 50K
3. Train 2-phase (synthetic → synthetic+real)
4. Eval + self-train nếu cần
