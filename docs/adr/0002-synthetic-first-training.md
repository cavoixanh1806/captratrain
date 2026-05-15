# ADR-0002: Synthetic-First Training Strategy

## Status
Accepted

## Context
Dataset ban đầu: 500 ảnh real, **không có label**.
- Không thể train supervised end-to-end
- Manual label 500 ảnh tốn thời gian, error-prone
- Pretrained models (ddddocr) fail 100% trên domain này

## Decision
Pipeline 4-phase:
1. **Phase 1**: Train trên real data (sau khi có labels từ manual labeling tool)
2. **Phase 2**: Sinh 50K–100K synthetic từ calibrated renderer
3. **Phase 3**: Self-training — pseudo-labels từ confident predictions (>0.95)
4. **Phase 4**: Fine-tune và deploy

## Consequences

### Positive
- Scale training data lên 100K+ samples
- Chỉ cần manual label ~50–100 hard cases (thay vì 500)
- Domain adaptation qua self-training

### Risk
- Quality phụ thuộc synthetic renderer matching real captcha
- Mitigation: Calibrate renderer từ analysis của 754 real ảnh (background, font, overlap patterns)

## Calibration Parameters
- Background: BGR avg (160, 157, 156), 70% pure gray
- Overlap: 55% dense merge, 25% medium, 20% light
- Rotation: 65% [-5°, 5°], 21% [-15°, 15°]
- Font: 60% bold, 40% regular, 12 candidates

## References
- `research_minecraft_map_captcha_20260515.md`
- `synthetic_renderer.py` calibration code
