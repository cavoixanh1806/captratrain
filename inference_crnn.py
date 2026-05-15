"""
inference_crnn.py
==================
Inference cho CRNN+CTC model.

Replace inference.py (TrOCR) với pipeline đơn giản hơn:
    Image → resize 64×256 → CRNN → CTC greedy decode → 5-char string

Usage CLI:
    python inference_crnn.py data/map_00001.png

Usage code:
    from inference_crnn import CRNNCaptchaSolver
    solver = CRNNCaptchaSolver()
    text = solver.solve("path/to/captcha.png")
    print(text)  # "4KTN9"

    # Batch
    results = solver.solve_batch(["a.png", "b.png"])

    # Với confidence
    text, conf = solver.solve_with_confidence("path/to/captcha.png")
    print(f"{text} ({conf:.2%})")
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from crnn_model import (
    CAPTCHA_CHARSET,
    CRNN,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    decode_greedy,
    decode_greedy_with_confidence,
    load_crnn,
)
from dataset_crnn import _MEAN, _STD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT: str = "captcha_crnn_model.pth"
EXPECTED_LENGTH: int = 5


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR → normalized tensor (3, H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR,
    )
    arr = resized.astype(np.float32) / 255.0
    for i, (m, s) in enumerate(zip(_MEAN, _STD)):
        arr[..., i] = (arr[..., i] - m) / s
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _enforce_length(text: str, target: int = EXPECTED_LENGTH) -> str:
    """Đảm bảo output đúng length=5.

    CTC greedy có thể đôi khi sinh 4 hoặc 6 chars. Strategy:
        - Nếu dài hơn: cắt giữ 5 chars đầu.
        - Nếu ngắn hơn: pad bằng ký tự đầu tiên trong charset (placeholder).
          (Sẽ luôn sai, nhưng ít sai hơn để CER không tăng quá).
    """
    if len(text) == target:
        return text
    if len(text) > target:
        return text[:target]
    # Pad với ký tự đầu tiên trong charset
    return text + CAPTCHA_CHARSET[0] * (target - len(text))


class CRNNCaptchaSolver:
    """Solver dùng CRNN model.

    Args:
        checkpoint_path: path đến .pth file (lưu bằng `save_crnn`).
        device: "cuda" | "cpu" | None (auto detect).
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint không tồn tại: {ckpt_path}\n"
                f"Hãy train trước: python train_crnn.py"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Device: {self.device}")

        logger.info(f"Loading CRNN from: {ckpt_path}")
        self.model: CRNN = load_crnn(str(ckpt_path), device=str(self.device))
        logger.info(f"Model params: {self.model.count_parameters():,}")
        logger.info("✅ Solver ready.")

    def _read_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image không tồn tại: {path}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Không đọc được ảnh: {path}")
        return bgr

    def solve(self, image_path: str | Path) -> str:
        """Predict 1 ảnh, trả về string 5 chars."""
        bgr = self._read_image(image_path)
        tensor = _preprocess(bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        pred = decode_greedy(logits)[0]
        return _enforce_length(pred)

    def solve_with_confidence(
        self,
        image_path: str | Path,
    ) -> tuple[str, float]:
        """Predict 1 ảnh + average confidence per char.

        Returns:
            (text, confidence) — text length=5, confidence in [0,1].
        """
        bgr = self._read_image(image_path)
        tensor = _preprocess(bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        preds, confs = decode_greedy_with_confidence(logits)
        return _enforce_length(preds[0]), confs[0]

    def solve_batch(
        self,
        image_paths: list[str | Path],
        batch_size: int = 64,
    ) -> list[str]:
        """Predict batch.

        Args:
            image_paths: list path.
            batch_size: chia thành các batch nhỏ để tránh OOM.
        """
        results: list[str] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                bgr = self._read_image(p)
                tensors.append(_preprocess(bgr))
            batch = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
            preds = decode_greedy(logits)
            results.extend(_enforce_length(p) for p in preds)
        return results

    def solve_batch_with_confidence(
        self,
        image_paths: list[str | Path],
        batch_size: int = 64,
    ) -> list[tuple[str, float]]:
        """Batch predict + confidence."""
        results: list[tuple[str, float]] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                bgr = self._read_image(p)
                tensors.append(_preprocess(bgr))
            batch = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
            preds, confs = decode_greedy_with_confidence(logits)
            for p, c in zip(preds, confs):
                results.append((_enforce_length(p), c))
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CRNN CAPTCHA inference")
    parser.add_argument("image", type=str, help="Path to CAPTCHA image")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    solver = CRNNCaptchaSolver(args.checkpoint, args.device)
    text, conf = solver.solve_with_confidence(args.image)
    print(f"Result: {text}  (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
