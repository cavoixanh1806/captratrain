"""
eval_crnn.py
=============
Evaluate CRNN model trên toàn bộ real CAPTCHA.

In ra:
    - Exact Match Accuracy (% ảnh đoán đúng 5/5)
    - CER (Character Error Rate)
    - Per-position accuracy
    - Top-10 confusions
    - Distribution of confidence scores
    - Verdict (PASS/FAIL theo target ≥ 90%)

Usage:
    python eval_crnn.py
    python eval_crnn.py --batch-size 128
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from inference_crnn import CRNNCaptchaSolver, DEFAULT_CHECKPOINT
from train_crnn import _edit_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(
    checkpoint: str = DEFAULT_CHECKPOINT,
    metadata_path: str = "data/metadata.csv",
    image_dir: str = "data",
    batch_size: int = 64,
    json_out: "str | Path | None" = None,
) -> dict:
    """Run evaluation.

    Returns:
        Dict tóm tắt metrics. When ``json_out`` is provided, the same dict
        (plus a ``confusion_matrix`` map and a ``low_confidence_wrongs``
        list) is also written to that path as JSON.
    """
    metadata_path = Path(metadata_path)
    image_dir = Path(image_dir)

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return {}

    df = pd.read_csv(metadata_path, dtype=str).dropna()
    df["text"] = df["text"].str.strip().str.upper()
    df = df[df["text"].str.len() == 5].reset_index(drop=True)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Checkpoint: {checkpoint}")

    solver = CRNNCaptchaSolver(checkpoint)

    image_paths = [str(image_dir / fn) for fn in df["filename"]]
    labels = df["text"].tolist()

    logger.info("Predicting...")
    results = solver.solve_batch_with_confidence(
        image_paths, batch_size=batch_size,
    )
    preds = [r[0] for r in results]
    confs = [r[1] for r in results]

    # ── Metrics ──────────────────────────────────────────────────────────────
    exact_correct = sum(1 for p, l in zip(preds, labels) if p == l)
    exact_acc = exact_correct / len(labels)

    total_dist = sum(_edit_distance(p, l) for p, l in zip(preds, labels))
    total_chars = sum(max(len(l), 1) for l in labels)
    cer = total_dist / max(total_chars, 1)

    # Per-position accuracy
    pos_correct = [0] * 5
    pos_total = [0] * 5
    for pred, label in zip(preds, labels):
        for i in range(min(len(pred), len(label), 5)):
            if pred[i] == label[i]:
                pos_correct[i] += 1
            pos_total[i] += 1
    pos_acc = [c / max(t, 1) for c, t in zip(pos_correct, pos_total)]

    # Confusion top
    confusions: Counter[str] = Counter()
    for pred, label in zip(preds, labels):
        for i in range(min(len(pred), len(label), 5)):
            if pred[i] != label[i]:
                confusions[f"{label[i]} → {pred[i]}"] += 1

    # Confidence distribution
    avg_conf = sum(confs) / max(len(confs), 1)
    conf_correct = [c for p, l, c in zip(preds, labels, confs) if p == l]
    conf_wrong = [c for p, l, c in zip(preds, labels, confs) if p != l]

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("CRNN EVALUATION RESULTS")
    print("=" * 64)
    print(f"Total samples:       {len(labels)}")
    print(f"Exact match correct: {exact_correct}")
    print(f"Exact match acc:     {exact_acc * 100:6.2f}%")
    print(f"CER:                 {cer * 100:6.2f}%")
    print(f"Avg confidence:      {avg_conf * 100:6.2f}%")
    if conf_correct:
        print(f"  Avg conf (correct): {sum(conf_correct)/len(conf_correct)*100:6.2f}%")
    if conf_wrong:
        print(f"  Avg conf (wrong):   {sum(conf_wrong)/len(conf_wrong)*100:6.2f}%")
    print()
    print("Per-position accuracy:")
    for i, acc in enumerate(pos_acc):
        bar = "█" * int(acc * 30)
        print(f"  Position {i + 1}: {acc * 100:6.2f}%  {bar}")
    print()
    if confusions:
        print(f"Top 10 confusions ({sum(confusions.values())} total mistakes):")
        for confusion, count in confusions.most_common(10):
            print(f"  {confusion}: {count}")

    # Wrong samples by lowest confidence
    wrong_samples = [
        (fn, l, p, c)
        for fn, l, p, c in zip(df["filename"], labels, preds, confs)
        if p != l
    ]
    wrong_samples.sort(key=lambda x: x[3])  # lowest conf first
    if wrong_samples:
        print()
        print("10 lowest-confidence WRONG predictions:")
        for fn, l, p, c in wrong_samples[:10]:
            print(f"  {fn}: label={l}  pred={p}  conf={c*100:5.2f}%")

    # Verdict
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    if exact_acc >= 0.90:
        print(f"[EXCELLENT] Exact match {exact_acc*100:.1f}% ≥ 90% — ACHIEVED TARGET")
    elif exact_acc >= 0.80:
        print(f"[GOOD] Exact match {exact_acc*100:.1f}% — close to target, fine-tune more.")
    elif exact_acc >= 0.50:
        print(f"[OK] Exact match {exact_acc*100:.1f}% — model works, needs more data/epochs.")
    else:
        print(f"[FAIL] Exact match {exact_acc*100:.1f}% — review pipeline.")

    # ── Structured outputs (return + optional JSON dump) ─────────────────────
    # 2-D confusion map: {true_char: {pred_char: count, ...}, ...}
    confusion_matrix: dict[str, dict[str, int]] = {}
    for pred, label in zip(preds, labels):
        for i in range(min(len(pred), len(label), 5)):
            t_char, p_char = label[i], pred[i]
            if t_char != p_char:
                confusion_matrix.setdefault(t_char, {})
                confusion_matrix[t_char][p_char] = (
                    confusion_matrix[t_char].get(p_char, 0) + 1
                )

    low_confidence_wrongs = [
        {"file": fn, "true": l, "pred": p, "confidence": float(c)}
        for fn, l, p, c in wrong_samples[:50]
    ]

    summary = {
        "total": len(labels),
        "exact_match": exact_acc,
        "cer": cer,
        "per_position": pos_acc,
        "confusions": dict(confusions.most_common(20)),
        "confusion_matrix": confusion_matrix,
        "avg_confidence": avg_conf,
        "low_confidence_wrongs": low_confidence_wrongs,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "metadata_path": str(metadata_path),
    }

    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Eval JSON written: {json_path}")

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRNN on real CAPTCHA")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--json-out", default=None,
        help=(
            "Optional path to dump the full evaluation summary as JSON "
            "(includes confusion_matrix and low_confidence_wrongs)."
        ),
    )
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        metadata_path=args.metadata,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
