"""
evaluate.py
===========
Danh gia model TrOCR tren toan bo 500 anh real CAPTCHA.

Tinh:
- Exact Match Accuracy: % anh dung hoan toan 5/5 ky tu
- CER (Character Error Rate): % ky tu sai
- Per-character accuracy
- Confusion: nhung ky tu hay bi sai

Usage:
    python evaluate.py
    python evaluate.py --model-dir ./captcha_trocr_model
"""

import argparse
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import evaluate as hf_evaluate

from inference import CaptchaSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_dir: str = "./captcha_trocr_model",
    metadata_path: str = "data/metadata.csv",
    image_dir: str = "data",
    batch_size: int = 16,
) -> dict:
    """Danh gia model tren toan bo metadata.csv.

    Args:
        model_dir: Duong dan model TrOCR.
        metadata_path: File CSV chua filename + text label.
        image_dir: Thu muc anh.
        batch_size: Batch size cho inference.

    Returns:
        Dict cac metric.
    """
    metadata_path = Path(metadata_path)
    image_dir = Path(image_dir)

    if not metadata_path.exists():
        logger.error(f"Khong tim thay {metadata_path}")
        return {}

    df = pd.read_csv(metadata_path, dtype=str).dropna()
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    logger.info(f"Total: {len(df)} anh de evaluate")
    logger.info(f"Model: {model_dir}")

    # Load model
    solver = CaptchaSolver(model_dir=model_dir)

    # Batch inference
    image_paths = [image_dir / row["filename"] for _, row in df.iterrows()]
    labels = [row["text"].strip().upper() for _, row in df.iterrows()]

    logger.info("Bat dau predict...")
    predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = solver.solve_batch(batch)
        predictions.extend(batch_results)
        if (i + batch_size) % 100 == 0:
            logger.info(f"  Predicted {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    # === Calculate metrics ===
    # 1. Exact match
    exact_correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    exact_accuracy = exact_correct / len(labels)

    # 2. CER
    cer_metric = hf_evaluate.load("cer")
    cer = cer_metric.compute(predictions=predictions, references=labels)

    # 3. Per-position accuracy
    pos_correct = [0] * 5
    pos_total = [0] * 5
    for pred, label in zip(predictions, labels):
        for i in range(min(len(pred), len(label), 5)):
            if pred[i] == label[i]:
                pos_correct[i] += 1
            pos_total[i] += 1
    pos_accuracy = [c / max(t, 1) for c, t in zip(pos_correct, pos_total)]

    # 4. Common confusions (top 10)
    confusions = Counter()
    for pred, label in zip(predictions, labels):
        for i in range(min(len(pred), len(label), 5)):
            if pred[i] != label[i]:
                confusions[f"{label[i]} → {pred[i]}"] += 1

    # === Report ===
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:       {len(labels)}")
    print(f"Exact match correct: {exact_correct}")
    print(f"Exact match acc:     {exact_accuracy * 100:.2f}%")
    print(f"CER:                 {cer * 100:.2f}%")
    print()
    print("Per-position accuracy:")
    for i, acc in enumerate(pos_accuracy):
        print(f"  Position {i + 1}: {acc * 100:.2f}%")
    print()
    print("Top 10 confusions (label → predicted):")
    for confusion, count in confusions.most_common(10):
        print(f"  {confusion}: {count}")

    # === Verdict ===
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if exact_accuracy >= 0.90:
        print(f"[EXCELLENT] Exact match {exact_accuracy*100:.1f}% >= 90% — DAT MUC TIEU!")
    elif exact_accuracy >= 0.80:
        print(f"[GOOD] Exact match {exact_accuracy*100:.1f}% — gan dat 90%, can tinh chinh them.")
    elif exact_accuracy >= 0.50:
        print(f"[OK] Exact match {exact_accuracy*100:.1f}% — Model hoat dong, can train them.")
    else:
        print(f"[FAIL] Exact match {exact_accuracy*100:.1f}% — Can xem lai pipeline.")

    return {
        "total": len(labels),
        "exact_match": exact_accuracy,
        "cer": cer,
        "per_position": pos_accuracy,
        "confusions": dict(confusions.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrOCR on real CAPTCHA")
    parser.add_argument("--model-dir", default="./captcha_trocr_model")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        metadata_path=args.metadata,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
