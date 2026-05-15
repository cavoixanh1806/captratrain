# CONTEXT.md — CAPTCHA Solver Domain Glossary

## Core Concepts

### captcha
Ảnh 128×128 RGB chứa 5 ký tự cần nhận diện. Được render trên nền Minecraft map terrain. Filename: `map_<index>.png`.

### label
Chuỗi 5 ký tự là kết quả của captcha. Ví dụ: `"4KTN9"`. Lưu trong `metadata.csv` mapping filename → label.

### charset
Tập 24 ký tự được phép trong captcha: `ACDEFHJKLMNPQRTUVWXY3479`. Các cặp dễ nhầm đã loại: `O/0`, `I/1`, `S/5`, `B/8`, `G/6`, `Z/2`.

### synthetic
Ảnh captcha được sinh tự động từ `synthetic_renderer.py`. Dùng Minecraft font + calibrated backgrounds để match real data. Scale: 50K–100K ảnh.

### real
Ảnh captcha thật từ Minecraft (754 ảnh). Dùng cho train và eval chính.

## Model Architecture

### crnn
Convolutional RNN — CNN backbone (7 blocks) + BiLSTM (2 layers, hidden=128). ~2.18M params. Input 64×320 → Output (T=79, B, 25).

### ctc
Connectionist Temporal Classification. Loss function và decoder cho sequence alignment. Blank token = index 0.

### exact_match
Metric: % predictions đúng cả 5 ký tự. Target ≥ 90%.

### cer
Character Error Rate. Target ≤ 10%.

## Pipeline Phases

### phase_1_train
Train CRNN trên real data (754 ảnh). Val split 15%. Baseline round.

### phase_2_synthetic
Sinh 50K–100K synthetic từ calibrated renderer. Train augmentation.

### phase_3_self_train
Self-training: pseudo-labels từ confident predictions (>0.95) trên real data. Fine-tune round 2.

### phase_4_deploy
Export ONNX → deploy via `ddddocr` hoặc onnxruntime.

## File Naming

- `map_<5-digit>.png` — Real captcha images
- `map_<label>_<hash>.png` — Synthetic generated images
- `captcha_crnn_last.pth` — PyTorch trained weights
- `captcha_crnn_model.onnx` — ONNX export for deployment
