# Bugfix Requirements Document

## Introduction

Sau 200 epochs train CRNN+CTC trên 754 ảnh thật (`data/metadata.csv`) với GPU
RTX 3060 8GB / CUDA 12.8, pipeline thất bại trên hai trục cùng lúc:

1. **Correctness (CTC collapse + overfit on augmentation)** — `train_log.txt`
   ghi nhận `eval_exact_match = 0.000` ở **mọi** epoch (1 → 200). Train loss
   giảm `44.39 → 2.58` nhưng eval loss đi ngược lại `41.12 → 3.82`, vượt cả
   ngưỡng "predict toàn blank" (~`log(25) = 3.22`). Model memorize augmentation
   patterns trên train, không decode được captcha thật trên val.
2. **Performance (CPU-bound pipeline)** — quan sát `nvidia-smi` cho thấy GPU
   util ~50% trong khi 12 luồng CPU bão hòa 100%. `train_crnn.py` cố định
   `num_workers = 0` (kèm comment "Windows pickle issue với albumentations")
   và augmentation albumentations chạy đồng bộ trên CPU trong thread chính
   của DataLoader, nên GPU phải chờ batch.

Bug được kích hoạt bởi tổ hợp augmentation hiện tại trong `dataset_crnn.py`
(Affine `rotate=(-12,12)` + `shear=(-5,5)` + Perspective `scale=(0.02,0.08)`
+ ColorJitter mạnh + GaussNoise + Blur + CoarseDropout, áp dụng `p=0.6`),
hyperparam trong `train_crnn.py` (`DEFAULT_LR=1e-3`, `WARMUP_STEPS=200`),
cấu hình kiến trúc trong `crnn_model.py` (`hidden_size=128`, ~2.18M params),
và `num_workers=0` cứng trong DataLoader.

Phạm vi bugfix:
- **Có**: chạm vào `dataset_crnn.py`, `train_crnn.py`, `crnn_model.py` để
  khôi phục khả năng học của CRNN+CTC trên 754 ảnh thật và để pipeline khai
  thác đúng GPU trên Windows.
- **Có**: hòa giải doc/code drift đã ghi trong `docs/codebase/CONCERNS.md`
  (`hidden_size` 128 vs 256, default epochs 200 vs 50) bằng một nguồn sự
  thật duy nhất.
- **Không**: sinh synthetic data — bị loại trừ rõ ràng theo yêu cầu của
  user. Mọi đề xuất "sinh 50K-100K synthetic" trong
  `docs/research_strategy_20260515.md` không thuộc phạm vi bugfix này.
- **Không**: chạm hardware target khác ngoài i5-12400F + RTX 3060 8GB +
  16GB DDR4 + Windows / CUDA 12.8.
- **Không**: thay đổi charset (24 chars), input shape (64×320), giao thức
  ONNX, hoặc API của `CRNNCaptchaSolver`.

## Bug Analysis

### Current Behavior (Defect)

Mỗi tiêu chí dưới đây gắn với số liệu cụ thể trong `train_log.txt` (run
bắt đầu `2026-05-15 18:32:01`, kết thúc `2026-05-15 18:38:35`, 200 epoch,
20 train-batch/epoch, 4 val-batch/epoch, train=641 / val=113, seed=42).

1.1 WHEN train CRNN+CTC trên 754 ảnh thật trong `data/metadata.csv`
(train=641, val=113) với augmentation hiện tại của
`dataset_crnn._build_albu_aug(strong=True)` (Affine `rotate=(-12, 12)`,
`shear=(-5, 5)`, `translate_percent=(-0.06, 0.06)`, `scale=(0.85, 1.15)`,
`p=0.6`; Perspective `scale=(0.02, 0.08)`, `p=0.3`;
RandomBrightnessContrast `p=0.5`; HueSaturationValue `p=0.4`;
GaussNoise `p=0.4`; GaussianBlur/MotionBlur `p=0.25`; CoarseDropout
`p=0.2`), THEN `eval_exact_match` báo cáo trong `train_log.txt` = `0.000`
ở **mọi** epoch được log (epoch 1, 2, …, 200 — lấy mẫu epoch 1, 10, 50,
100, 150, 200 đều cho `0.000`), và dòng tổng kết cuối log ghi
`[DONE] Best val_exact_match=0.0000 at epoch 0`.

1.2 WHEN warmup hoàn tất (`learning_rate = 1.000e-03` đạt ở epoch 10) và
train tiếp tục, THEN `train_loss` mắc kẹt ở vùng `log(NUM_CLASSES) =
log(25) ≈ 3.219` trong hàng chục epoch — cụ thể `train_loss = 3.5138`
(epoch 10), `3.3709` (epoch 25), `3.3293` (epoch 50), chứng tỏ network
đang predict gần như đồng nhất / blank.

1.3 WHEN train tiếp diễn sau khi `eval_loss` chạm đáy `3.388` (epoch 18,
xác nhận lại ở epoch 21 = `3.388`), THEN `eval_loss` tăng đơn điệu trong
khi `train_loss` giảm — số liệu trong log: `eval_loss/train_loss` =
`3.426/3.3293` (e50), `3.449/3.2650` (e75), `3.516/3.1573` (e100),
`3.611/2.9531` (e125), `3.732/2.7531` (e150), `3.799/2.6069` (e175),
`3.823/2.5765` (e200). Khoảng cách `eval_loss − train_loss` mở rộng từ
`+0.10` (e50) đến `+1.25` nats (e200), và `eval_loss = 3.823` cuối cùng
**vượt** `log(25) ≈ 3.219` — model trên val còn tệ hơn output uniform
blank, đặc trưng overfit lên augmentation chứ không học mapping image →
text.

1.4 WHEN evaluate trên val split 113 ảnh real, THEN `eval_cer` không bao
giờ giảm xuống dưới `0.919` trong toàn bộ 200 epoch (trị thấp nhất
quan sát: `0.919` ở epoch 33; trị cuối cùng: `0.949` ở epoch 200) —
trung bình mỗi captcha 5 ký tự chỉ phục hồi được dưới 0.5 ký tự đúng,
quá xa ngưỡng cần để có một exact match nào.

1.5 WHEN một epoch chạy trên target i5-12400F + RTX 3060 + 16GB DDR4 +
Windows + CUDA 12.8 với cấu hình hiện tại của `train_crnn.main`
(`num_workers = 0` hard-code kèm comment "Windows: 0 để tránh pickle
issue với albumentations" và toàn bộ pipeline albumentations chạy đồng
bộ trong `__getitem__` trên thread chính), THEN GPU utilization quan sát
qua `nvidia-smi` trung bình ≈ 50% trong khi 12 luồng CPU bão hòa gần
100% — DataLoader là bottleneck và GPU idle chờ batch (gián tiếp xác
nhận bởi throughput log: ~10.7 train-it/s, ~1.8s/epoch trên 20 batch).

1.6 WHEN một người đọc tham chiếu chéo doc và code, THEN doc
(`README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`,
`docs/adr/0001-crnn-ctc-over-softmax.md`) ghi BiLSTM `hidden_size = 256`
và default `Epochs = 50`, trong khi code (`crnn_model.CRNN.__init__`)
ghi `hidden_size: int = 128` và `train_crnn.DEFAULT_EPOCHS = 200` —
mâu thuẫn được khẳng định bởi `train_log.txt` (`CRNN params: 2,186,553`,
`Training for 200 epochs`). Con số "~2.18M params" doc đang quote chỉ
khớp với `hidden=128`, không khớp với `hidden=256`.

### Expected Behavior (Correct)

Mỗi tiêu chí được thiết kế để có thể kiểm chứng độc lập trên cùng môi
trường đã chạy run lỗi (754 ảnh real, 113 ảnh val, seed=42, Windows +
RTX 3060 + i5-12400F).

2.1 WHEN train CRNN+CTC end-to-end trên 754 ảnh real (`data/metadata.csv`,
train=641 / val=113, `seed=42`) với pipeline đã sửa, THEN tại ít nhất
một checkpoint được lưu trong run đó, hệ thống SHALL đạt
`eval_exact_match ≥ 0.30` trên val split 113 ảnh real (≥ 34 ảnh được
decode đúng cả 5/5 ký tự).

   **Rationale (vì sao 0.30 chứ không 0.50):** run lỗi có
   `eval_exact_match = 0.000` ở mọi epoch và `eval_cer ≥ 0.919` xuyên
   suốt — do đó bất kỳ giá trị `> 0` đã là cải thiện step-function
   chứng minh CTC không còn collapse. Trên 113 ảnh val, ngưỡng `0.30`
   tương đương ≥ 34 ảnh đúng và độ lệch chuẩn 1-σ ≈ ±4 điểm phần trăm
   (`sqrt(0.3 × 0.7 / 113) ≈ 0.043`) — đủ cao để phân biệt thống kê
   với 0, đủ thấp để trung thực với 641 ảnh train không có synthetic.
   Đặt mốc `≥ 0.50` không có cơ sở khi corpus chỉ 754 ảnh và phạm vi
   bugfix loại trừ synthetic.

2.2 WHEN train kết thúc theo budget cấu hình, THEN tại checkpoint tốt
nhất `eval_loss` SHALL **thấp hơn nghiêm ngặt** đáy `3.388` đã ghi nhận
trong run lỗi (epoch 18 trong `train_log.txt`) AND **thấp hơn nghiêm
ngặt** `log(NUM_CLASSES) = log(25) ≈ 3.219` — nghĩa là model SHALL
generalise tốt hơn mức "predict uniform blank" trên val.

2.3 WHEN train tiếp tục sau warmup, THEN tại checkpoint tốt nhất gap
`eval_loss − train_loss` SHALL `≤ 0.50` nats (so với `+1.25` nats ở
epoch 200 trong run lỗi) — chấm dứt mode phân kỳ
`train_loss ↓` đồng thời `eval_loss ↑` xuất hiện từ epoch 18 đến epoch
200 trong `train_log.txt`.

2.4 WHEN một epoch ở chế độ steady-state chạy trên target i5-12400F +
RTX 3060 + 16GB DDR4 + Windows + CUDA 12.8 sau khi DataLoader đã warm-up
(bỏ epoch đầu tiên), THEN GPU utilization lấy mẫu bằng
`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000`
trong cửa sổ liên tục 30 giây ở giữa training (loại trừ epoch đầu, eval
step, và bước ONNX export cuối) SHALL trung bình **≥ 80%**, và trên
hardware target DataLoader SHALL không còn là bottleneck.

2.5 WHEN người đọc tra cứu `hidden_size` BiLSTM, default epoch count,
default batch size, hoặc param count, THEN source code
(`crnn_model.py` và `train_crnn.py`) SHALL là **nguồn sự thật duy nhất
(canonical source of truth)**, và mọi document trong repo nhắc đến các
giá trị này (`README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`,
`docs/adr/0001-crnn-ctc-over-softmax.md`, `docs/codebase/ARCHITECTURE.md`,
`docs/codebase/CONCERNS.md`) SHALL hoặc tham chiếu hằng số theo tên,
hoặc quote đúng giá trị đã định nghĩa trong code; con số
parameter-count quote trong doc SHALL bằng đúng giá trị
`CRNN().count_parameters()` in ra trên cấu hình canonical đó (run lỗi
in `2,186,553` cho `hidden=128`).

### Unchanged Behavior (Regression Prevention)

3.1 WHEN nạp một ảnh CAPTCHA bất kỳ qua `inference_crnn.CRNNCaptchaSolver`
hoặc qua ONNX runtime, THEN charset SHALL CONTINUE TO là đúng 24 ký tự
`"ACDEFHJKLMNPQRTUVWXY3479"` với `CTC_BLANK_INDEX = 0` và
`NUM_CLASSES = 25`, không thêm/bớt/đổi thứ tự ký tự nào.

3.2 WHEN ảnh đầu vào được tiền xử lý cho CRNN, THEN input shape SHALL
CONTINUE TO là `(B, 3, 64, 320)` (1:5 aspect ratio, ImageNet
mean/std normalize), và mọi đường preprocessing trong `dataset_crnn` +
`inference_crnn._preprocess` SHALL CONTINUE TO sản sinh cùng tensor
shape/dtype.

3.3 WHEN chạy `train_crnn.main` đến cuối, THEN file ONNX export
(`captcha_crnn_model.onnx`, opset 14, dynamic batch axis) SHALL CONTINUE
TO được sinh ra với cùng input/output names ("input", "logits") và cùng
contract như mô tả trong `README.md` mục "Deploy với ONNX runtime".

3.4 WHEN người dùng chạy `python train_crnn.py --resume`, THEN flow
resume từ `captcha_crnn_last.pth` SHALL CONTINUE TO khôi phục
`state_dict`, `optimizer`, `scheduler`, `scaler`, `epoch`, `best_val_em`,
`best_epoch` và tiếp tục training từ đúng epoch kế tiếp.

3.5 WHEN người dùng chạy lệnh train mặc định trên Windows
(`python train_crnn.py`) trong môi trường có albumentations và CUDA 12.8,
THEN training SHALL CONTINUE TO khởi động không lỗi pickle/spawn và
không yêu cầu tương tác thủ công ngoài việc chạy lệnh — pipeline phải
tương thích Windows cho cả `num_workers = 0` và `num_workers > 0` (nếu
fix chuyển sang giá trị > 0).

3.6 WHEN một ảnh được decode qua `crnn_model.decode_greedy` hoặc
`decode_greedy_with_confidence`, THEN ngữ nghĩa decode SHALL CONTINUE TO
là CTC greedy chuẩn (per-step argmax → collapse repeats → drop blanks)
và `inference_crnn._enforce_length` SHALL CONTINUE TO trả output đúng
length = 5 (cắt nếu dài, pad bằng `CAPTCHA_CHARSET[0]` nếu ngắn).

3.7 WHEN training chia train/val split, THEN logic split của
`dataset_crnn.create_crnn_datasets` (15% val real, `seed = 42`,
val luôn lấy từ real) SHALL CONTINUE TO áp dụng để metrics so sánh được
giữa các lần train.

3.8 WHEN chạy `python eval_crnn.py` sau khi train xong, THEN
`CRNNCaptchaSolver` API (`solve`, `solve_with_confidence`, `solve_batch`,
`solve_batch_with_confidence`) và format output của
`eval_crnn.evaluate` (exact_match, CER, per-position accuracy, top-10
confusions, verdict) SHALL CONTINUE TO không đổi.

3.9 WHEN bugfix được áp dụng, THEN nguồn data train SHALL CONTINUE TO
chỉ là 754 ảnh thật trong `data/metadata.csv` — không thêm synthetic,
không gọi `generate_synthetic_crnn.py`, và pipeline lệnh mặc định
(`python train_crnn.py`) không phụ thuộc bất kỳ data synthetic nào.

3.10 WHEN bugfix được áp dụng, THEN hardware target SHALL CONTINUE TO
là Windows + i5-12400F (6P+0E, 12T) + 16GB DDR4 + RTX 3060 8GB
(CUDA 12.8) — VRAM peak SHALL không vượt 8GB và RAM peak SHALL không
vượt 16GB ở batch size mặc định.
