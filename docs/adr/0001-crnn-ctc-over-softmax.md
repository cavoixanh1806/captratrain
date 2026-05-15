# ADR-0001: CRNN+CTC thay vì Multi-Head Softmax

## Status
Accepted

## Context
Captcha Minecraft Map có đặc điểm:
- Fixed length: 5 ký tự
- Ký tự chồng đè (overlapping) — boundaries không rõ
- Nền terrain phức tạp, noise lớn

Có 2 approach chính:
1. **Multi-head softmax**: 5 independent classifiers, mỗi head nhìn 1 vùng spatial
2. **CRNN+CTC**: Sequence model với alignment-free decoding

## Decision
Chọn **CRNN+CTC** với architecture:
- CNN backbone (7 blocks) + BiLSTM (2 layers, hidden=128 (see `crnn_model.CRNN` default))
- CTC loss + greedy decode
- Input resize 64×320 (aspect ratio 1:5)

## Implementation note (2026-05-15)

Bản draft ban đầu của ADR này quote `hidden_size = 2 * 128` (i.e. quoted
the bidirectional output width — `2 * hidden`), nhưng cấu hình canonical
thực tế đang chạy trong `crnn_model.CRNN.__init__` là `hidden_size=128`
per direction (không phải gấp đôi). Param count đã được verify bằng
`CRNN().count_parameters() == 2,186,553` và khớp với
`train_log.txt` (`CRNN params: 2,186,553`). FC layer cuối là
`Linear(2 * hidden_size, num_classes) = Linear(2*128, 25)` — con số
`2 * hidden = 2 * 128` (bidirectional width) trong các phiên bản trước
của doc đã bị nhầm với `hidden_size` per direction. Doc table trong
`README.md` / `PIPELINE_SUMMARY.md` / `CLAUDE.md` đã được sync về cùng
nguồn sự thật trong `crnn_model.CRNN.__init__` và
`train_crnn.DEFAULT_EPOCHS = 200`.

## Consequences

### Positive
- Xử lý ký tự chồng đè tốt hơn — CTC tự học alignment ngầm
- Không cần hard-coded character boundaries
- Mature literature, SOTA cho fixed-charset captcha

### Negative
- Phức tạp hơn multi-head (cần hiểu CTC decoding)
- Cần đủ timesteps (T=79) cho 5 ký tự — yêu cầu input kéo dãn ngang

## References
- [abhishekkrthakur/captcha-recognition-pytorch](https://github.com/abhishekkrthakur/captcha-recognition-pytorch)
- Graves et al. "Connectionist Temporal Classification"
