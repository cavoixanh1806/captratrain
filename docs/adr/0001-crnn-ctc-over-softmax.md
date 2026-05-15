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
- CNN backbone (7 blocks) + BiLSTM (2 layers, hidden=256)
- CTC loss + greedy decode
- Input resize 64×320 (aspect ratio 1:5)

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
