"""
crnn_model.py
=============
CRNN (CNN + BiLSTM + CTC) cho Minecraft Map CAPTCHA.

Theo research doc: CRNN+CTC là SOTA classic cho fixed-charset captcha,
hỗ trợ ký tự chồng đè qua CTC alignment ngầm.

Architecture (~8.72M params, count_parameters() == 8_718_937 with hidden_size=256):
    Input:    (B, 3, 64, 320) — RGB resized (1:5 aspect, kéo dãn ngang)
    CNN:      backbone 7 conv blocks, downsample H/16, W/4
                → (B, 512, h≈3, w=79)
    Pool:     adaptive avg pool height → (B, 512, 1, T=79) → permute
                → (T=79, B, 512)
    BiLSTM:   2 layers, hidden=256, dropout 0.2
                → (T=79, B, 512)  (bidir = 2*hidden)
    FC:       Linear(512 → 25)  (24 chars + 1 CTC blank)
    Output:   (T=79, B, 25)  — raw logits, log_softmax áp ở loss/decode

    T=79 → ~16 timesteps/char với 5 chars cố định, dư thừa cho CTC alignment.
    Tham khảo abhishekkrthakur/captcha-recognition-pytorch (75x300, ratio 1:4).

Charset: 24 ký tự (đã loại các cặp dễ nhầm như O/0, I/1, S/5, B/8, G/6, Z/2).
Blank token = index 0, ký tự = index 1..24.
"""

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# ─── Charset ──────────────────────────────────────────────────────────────────
CAPTCHA_CHARSET: str = "ACDEFHJKLMNPQRTUVWXY3479"
NUM_CLASSES: int = len(CAPTCHA_CHARSET) + 1  # +1 blank
CTC_BLANK_INDEX: int = 0
CHAR_TO_IDX: dict[str, int] = {c: i + 1 for i, c in enumerate(CAPTCHA_CHARSET)}
IDX_TO_CHAR: dict[int, str] = {i + 1: c for i, c in enumerate(CAPTCHA_CHARSET)}

# Input shape — 64x320 cho aspect ratio 1:5 (kéo dãn ngang).
# Lý do: captcha 128x128 vuông, khi resize thành 64x320 thì width gấp 5×
# chiều cao → mỗi ký tự chiếm nhiều pixels width hơn → CRNN có nhiều
# timesteps cho mỗi char → CTC alignment dễ học hơn. Tham khảo
# abhishekkrthakur/captcha-recognition-pytorch (75x300, ratio 1:4).
# Output T sau backbone ≈ 80 → 16 timesteps/char (rất dư cho CTC).
INPUT_HEIGHT: int = 64
INPUT_WIDTH: int = 320


class ConvBlock(nn.Module):
    """Conv → BN → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        pad: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# Canonical params: count_parameters() == 8_718_937 with hidden_size=256. Update PIPELINE_SUMMARY.md / README.md / CLAUDE.md if you change this.
class CRNN(nn.Module):
    """CRNN cho CAPTCHA OCR.

    Args:
        num_classes: số lớp output (24 chars + 1 blank = 25).
        hidden_size: hidden size của BiLSTM (mỗi chiều).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── CNN backbone ──────────────────────────────────────────────────────
        # Input: (B, 3, 64, 320). Goal: collapse H to ~1, keep W large for CTC.
        # Channel ladder: 64 → 128 → 256 → 256 → 512 → 512 → 512 (~5.5M params).
        self.cnn = nn.Sequential(
            ConvBlock(3, 64),                     # 64x320
            nn.MaxPool2d(2, 2),                   # 32x160
            ConvBlock(64, 128),                   # 32x160
            nn.MaxPool2d(2, 2),                   # 16x80
            ConvBlock(128, 256),                  # 16x80
            ConvBlock(256, 256),                  # 16x80
            nn.MaxPool2d((2, 1), (2, 1)),         # 8x80 (only H halved)
            ConvBlock(256, 512),                  # 8x80
            ConvBlock(512, 512),                  # 8x80
            nn.MaxPool2d((2, 1), (2, 1)),         # 4x80
            ConvBlock(512, 512, kernel=2, pad=0), # 3x79
        )

        # ── Sequence head ─────────────────────────────────────────────────────
        # Adaptive pool to height=1, keep width as timesteps
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        # 2-layer BiLSTM (~3.1M params)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        # Output: (T, B, 2*hidden_size)

        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) RGB image batch, normalized.

        Returns:
            logits: (T, B, num_classes) — raw logits, log_softmax in loss.
        """
        feat = self.cnn(x)               # (B, 512, h, w)
        feat = self.height_pool(feat)    # (B, 512, 1, w)
        feat = feat.squeeze(2)           # (B, 512, w)
        feat = feat.permute(2, 0, 1)     # (w, B, 512) — T=w

        rnn_out, _ = self.rnn(feat)      # (T, B, 2*hidden)
        logits = self.fc(rnn_out)        # (T, B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Encoding / Decoding ─────────────────────────────────────────────────────


def encode_label(text: str) -> list[int]:
    """Convert string label → list of int indices (1-based, 0 reserved for blank).

    Args:
        text: e.g. "4KTN9".

    Returns:
        List of indices, e.g. [22, 8, 16, 12, 24].

    Raises:
        ValueError: nếu text chứa ký tự ngoài charset.
    """
    text = text.upper()
    out = []
    for c in text:
        if c not in CHAR_TO_IDX:
            raise ValueError(
                f"Char {c!r} not in charset {CAPTCHA_CHARSET!r}. "
                f"Full label: {text!r}"
            )
        out.append(CHAR_TO_IDX[c])
    return out


def decode_greedy(logits: torch.Tensor) -> list[str]:
    """CTC greedy decode.

    Per-step argmax → collapse repeats → drop blanks.

    Args:
        logits: (T, B, C) raw logits hoặc log-probs.

    Returns:
        List[str], một chuỗi mỗi batch item.
    """
    # (T, B) → (B, T)
    pred = logits.argmax(dim=-1).permute(1, 0).cpu().tolist()
    decoded: list[str] = []
    for seq in pred:
        chars: list[str] = []
        prev = -1
        for idx in seq:
            if idx != prev:
                if idx != CTC_BLANK_INDEX and idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
                prev = idx
        decoded.append("".join(chars))
    return decoded


def decode_greedy_with_confidence(
    logits: torch.Tensor,
) -> tuple[list[str], list[float]]:
    """CTC greedy decode + average confidence per non-blank step.

    Args:
        logits: (T, B, C) raw logits.

    Returns:
        Tuple (decoded_strings, confidences in [0,1]).
    """
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    pred = log_probs.argmax(dim=-1)             # (T, B)
    pred_probs = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)  # (T, B)

    pred_t = pred.permute(1, 0).cpu().tolist()
    prob_t = pred_probs.permute(1, 0).cpu().tolist()

    decoded: list[str] = []
    confs: list[float] = []
    for seq, pseq in zip(pred_t, prob_t):
        chars: list[str] = []
        char_probs: list[float] = []
        prev = -1
        for idx, p in zip(seq, pseq):
            if idx != prev:
                if idx != CTC_BLANK_INDEX and idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
                    char_probs.append(float(p))
                prev = idx
        decoded.append("".join(chars))
        confs.append(sum(char_probs) / max(len(char_probs), 1))
    return decoded, confs


# ─── Save / Load ──────────────────────────────────────────────────────────────


def save_crnn(model: CRNN, path: str | Path) -> None:
    """Save state_dict + charset metadata."""
    payload = {
        "state_dict": model.state_dict(),
        "charset": CAPTCHA_CHARSET,
        "num_classes": model.num_classes,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
    }
    torch.save(payload, str(path))


def load_crnn(path: str | Path, device: str = "cpu") -> CRNN:
    """Load trained CRNN."""
    payload = torch.load(str(path), map_location=device)
    model = CRNN(num_classes=payload["num_classes"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


# ─── ONNX Export ──────────────────────────────────────────────────────────────


def export_onnx(model: CRNN, output_path: str | Path) -> None:
    """Export CRNN sang ONNX để deploy với onnxruntime hoặc ddddocr.

    Args:
        model: CRNN đã train.
        output_path: đường dẫn .onnx output.
    """
    model.eval()
    dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {1: "batch"},
        },
        opset_version=14,
    )


if __name__ == "__main__":
    # Smoke test
    model = CRNN()
    print(f"CRNN params: {model.count_parameters():,}")
    print(f"Charset: {CAPTCHA_CHARSET} ({len(CAPTCHA_CHARSET)} chars)")
    print(f"Num classes: {NUM_CLASSES} (+1 blank)")
    x = torch.randn(2, 3, INPUT_HEIGHT, INPUT_WIDTH)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (T, B, C)")
    print(f"Encode 'AKTN9' → {encode_label('4KTN9')}")
