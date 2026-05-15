"""
Property 1 — Bug Condition exploration test (BEFORE applying any fix).

This module encodes Property 1 from `.kiro/specs/crnn-ctc-collapse-fix/design.md`:

    "Trained CRNN+CTC Generalises To Real Validation Set And GPU is Saturated."

The test is written so that it MUST FAIL on UNFIXED code — failure is the
SUCCESS path because it confirms the bug exists. After the four-part fix
(A: dataset_crnn augmentation, B: train_crnn hyperparam + DataLoader,
C: crnn_model canonicalisation, D: doc/code reconciliation) is applied,
this same test file is re-run and is expected to PASS.

Layout:

    test_property1_smoke      — 20-epoch buggy-config training, asserts
                                smoke sub-thresholds (eval_em ≥ 0.05,
                                eval_loss < 3.0, gap ≤ 0.7).
    test_property1_full       — 200-epoch buggy-config training, asserts
                                the full Property 1 thresholds. Marked
                                @pytest.mark.slow.
    test_property1_gpu_util   — samples `nvidia-smi -lms 1000` for a
                                continuous 30 s window during a smoke
                                run; asserts mean GPU util ≥ 0.80.

Helper assertions (also from design "Exploratory Bug Condition Checking"):

    test_replay_log_confirms_bug       — parses `train_log.txt` from the
                                          historical buggy run; PASSES on
                                          unfixed code (confirms bug).
    test_doc_drift_confirms_bug        — greps the 6 doc files for
                                          `hidden=256` / `Epochs=50`;
                                          PASSES on unfixed code (≥ 1
                                          match exists).

Hardware target: Windows + i5-12400F + RTX 3060 8GB + CUDA 12.8.
On non-target hosts (no GPU / non-Windows / no nvidia-smi), the
training and GPU-util tests skip with a clear reason; the replay
and doc-drift helpers run everywhere.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Make the project root importable so we can `import train_crnn`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Constants from the design's `isBugCondition` pseudocode + Property 1
# thresholds. Keeping them named here so any threshold drift is immediately
# visible in PR diffs.
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES: int = 25
LOG_NUM_CLASSES: float = math.log(NUM_CLASSES)  # ≈ 3.2189

# Full Property 1 thresholds (post-fix expected behaviour).
FULL_MIN_EXACT_MATCH: float = 0.30
FULL_MAX_MIN_EVAL_LOSS_VS_LOG_K: float = LOG_NUM_CLASSES  # < log(25)
FULL_MAX_MIN_EVAL_LOSS_VS_BUGGY_FLOOR: float = 3.388  # also strictly less
FULL_MAX_MIN_GAP: float = 0.50  # eval_loss − train_loss

# Smoke sub-thresholds (relaxed for 20-epoch run).
SMOKE_MIN_EXACT_MATCH: float = 0.05
SMOKE_MAX_MIN_EVAL_LOSS: float = 3.0
SMOKE_MAX_MIN_GAP: float = 0.70

# GPU util threshold.
GPU_UTIL_WINDOW_SECONDS: int = 30
GPU_UTIL_MIN_MEAN: float = 0.80
GPU_UTIL_SAMPLE_PERIOD_MS: int = 1000

# Buggy-config training arguments (matches `isBugCondition` exactly).
BUGGY_CONFIG: dict[str, Any] = {
    "batch_size": 32,
    "lr": 1e-3,
    "use_synthetic": False,
    "use_real": True,
    "augment": True,
    "resume": False,
}
BUGGY_SEED: int = 42

DOCS_TO_CHECK_FOR_DRIFT: tuple[str, ...] = (
    "README.md",
    "PIPELINE_SUMMARY.md",
    "CLAUDE.md",
    "docs/adr/0001-crnn-ctc-over-softmax.md",
    "docs/codebase/ARCHITECTURE.md",
    "docs/codebase/CONCERNS.md",
)

DOC_DRIFT_PATTERN = re.compile(
    r"hidden(?:_size)?\s*=?\s*256|Epochs\s*=?\s*50",
    re.IGNORECASE,
)

# Patterns for parsing logger output / `print(eval_dict)` lines emitted by
# `train_crnn.train_one_epoch` / `train_crnn.main`.
TRAIN_LINE_RE = re.compile(
    r"\{'loss':\s*'([0-9.eE+-]+)'.*?'epoch':\s*'(\d+)'\}"
)
EVAL_LINE_RE = re.compile(
    r"\{'eval_loss':\s*'([0-9.eE+-]+)'.*?"
    r"'eval_exact_match':\s*'([0-9.eE+-]+)'.*?"
    r"'epoch':\s*'(\d+)'\}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware-target skip helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch  # noqa: WPS433 — local import on purpose
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _is_windows() -> bool:
    return os.name == "nt"


def _skip_if_not_hardware_target(require_smi: bool = False) -> None:
    """Skip cleanly if the host is not the bug's hardware target."""
    reasons: list[str] = []
    if not _is_windows():
        reasons.append("not Windows")
    if not _has_cuda():
        reasons.append("no CUDA-capable GPU available")
    if require_smi and not _has_nvidia_smi():
        reasons.append("nvidia-smi not on PATH")
    if reasons:
        pytest.skip(
            "Property 1 hardware-target preconditions not met: "
            + ", ".join(reasons)
            + ". This test is scoped to Windows + RTX 3060 + CUDA 12.8."
        )


def _require_dataset() -> None:
    """Skip if the real dataset is missing (needed to reproduce bug)."""
    meta = PROJECT_ROOT / "data" / "metadata.csv"
    if not meta.exists():
        pytest.skip(f"Required dataset not found: {meta}")


# ─────────────────────────────────────────────────────────────────────────────
# Capture helpers — run train_crnn.main and parse metrics from stdout/stderr
# ─────────────────────────────────────────────────────────────────────────────

class _OutputTee:
    """File-like that mirrors writes to an underlying stream and a buffer."""

    def __init__(self, underlying):
        self._underlying = underlying
        self.buffer: list[str] = []

    def write(self, s):  # type: ignore[no-untyped-def]
        self.buffer.append(s)
        try:
            self._underlying.write(s)
        except Exception:
            pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):  # type: ignore[no-untyped-def]
        try:
            self._underlying.flush()
        except Exception:
            pass

    def getvalue(self) -> str:
        return "".join(self.buffer)


def _run_buggy_training(epochs: int, seed: int = BUGGY_SEED) -> dict[str, list[float]]:
    """Run `train_crnn.main` on the exact buggy config and parse metrics.

    Returns a dict with three parallel lists: train_loss, eval_loss,
    eval_exact_match, indexed by epoch (1..N).
    """
    import torch
    import logging

    # Determinism: same seeds the bug was observed under.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    import train_crnn

    # Tee stdout + stderr so we capture both `tqdm.write` (training log_dict)
    # and `print(eval_dict)` (validation log_dict) lines.
    stdout_tee = _OutputTee(sys.stdout)
    stderr_tee = _OutputTee(sys.stderr)

    # Also capture the project logger (handlers usually go to stderr at INFO).
    log_buffer: list[str] = []

    class _BufHandler(logging.Handler):
        def emit(self, record):  # type: ignore[no-untyped-def]
            try:
                log_buffer.append(self.format(record))
            except Exception:
                pass

    buf_handler = _BufHandler(level=logging.INFO)
    buf_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(buf_handler)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_tee  # type: ignore[assignment]
    sys.stderr = stderr_tee  # type: ignore[assignment]
    try:
        train_crnn.main(
            epochs=epochs,
            batch_size=BUGGY_CONFIG["batch_size"],
            lr=BUGGY_CONFIG["lr"],
            use_synthetic=BUGGY_CONFIG["use_synthetic"],
            use_real=BUGGY_CONFIG["use_real"],
            augment=BUGGY_CONFIG["augment"],
            resume=BUGGY_CONFIG["resume"],
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        root_logger.removeHandler(buf_handler)

    captured = (
        stdout_tee.getvalue()
        + "\n"
        + stderr_tee.getvalue()
        + "\n"
        + "\n".join(log_buffer)
    )

    return _parse_metrics_from_output(captured)


def _parse_metrics_from_output(text: str) -> dict[str, list[float]]:
    """Parse `train_loss`, `eval_loss`, `eval_exact_match` keyed by epoch."""
    train_loss_by_epoch: dict[int, float] = {}
    eval_loss_by_epoch: dict[int, float] = {}
    eval_em_by_epoch: dict[int, float] = {}

    for m in TRAIN_LINE_RE.finditer(text):
        loss = float(m.group(1))
        epoch = int(m.group(2))
        train_loss_by_epoch[epoch] = loss

    for m in EVAL_LINE_RE.finditer(text):
        eval_loss = float(m.group(1))
        eval_em = float(m.group(2))
        epoch = int(m.group(3))
        eval_loss_by_epoch[epoch] = eval_loss
        eval_em_by_epoch[epoch] = eval_em

    epochs = sorted(set(train_loss_by_epoch) | set(eval_loss_by_epoch))
    return {
        "epochs": [float(e) for e in epochs],
        "train_loss": [train_loss_by_epoch[e] for e in epochs if e in train_loss_by_epoch],
        "eval_loss": [eval_loss_by_epoch[e] for e in epochs if e in eval_loss_by_epoch],
        "eval_exact_match": [eval_em_by_epoch[e] for e in epochs if e in eval_em_by_epoch],
    }


def _aggregate(metrics: dict[str, list[float]]) -> dict[str, float]:
    """Compute Property-1 aggregates from per-epoch metric lists."""
    eval_em = metrics["eval_exact_match"]
    eval_loss = metrics["eval_loss"]
    train_loss = metrics["train_loss"]

    # Pair train_loss / eval_loss by epoch where both are present.
    n_pairs = min(len(eval_loss), len(train_loss))
    gaps = [eval_loss[i] - train_loss[i] for i in range(n_pairs)]

    return {
        "max_eval_exact_match": max(eval_em) if eval_em else 0.0,
        "min_eval_loss": min(eval_loss) if eval_loss else float("inf"),
        "min_gap": min(gaps) if gaps else float("inf"),
        "final_eval_loss": eval_loss[-1] if eval_loss else float("inf"),
        "final_train_loss": train_loss[-1] if train_loss else float("inf"),
        "num_epochs_observed": len(eval_em),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Property 1 — smoke variant (20-epoch buggy-config training)
# ─────────────────────────────────────────────────────────────────────────────

def test_property1_smoke():
    """Property 1 smoke — 20 epochs on buggy config, assert smoke thresholds.

    EXPECTED ON UNFIXED CODE: FAILS (eval_exact_match stays 0.000,
    eval_loss stuck near log(25)). Failure proves the bug.

    EXPECTED ON FIXED CODE: PASSES.

    Validates: Requirements 2.1, 2.2, 2.3
    """
    _skip_if_not_hardware_target()
    _require_dataset()

    metrics = _run_buggy_training(epochs=20)
    agg = _aggregate(metrics)

    # Surface the counterexample in the assertion message so the failure
    # output documents the bug.
    cx = (
        f"Smoke run on data/metadata.csv (seed={BUGGY_SEED}, "
        f"{BUGGY_CONFIG}) for 20 epochs: "
        f"max_eval_exact_match={agg['max_eval_exact_match']:.4f}, "
        f"min_eval_loss={agg['min_eval_loss']:.4f}, "
        f"min(eval-train) gap={agg['min_gap']:.4f}, "
        f"final_eval_loss={agg['final_eval_loss']:.4f}, "
        f"log(NUM_CLASSES)={LOG_NUM_CLASSES:.4f}."
    )

    assert agg["max_eval_exact_match"] >= SMOKE_MIN_EXACT_MATCH, (
        f"max(eval_exact_match) = {agg['max_eval_exact_match']:.4f} < "
        f"{SMOKE_MIN_EXACT_MATCH:.4f}. {cx}"
    )
    assert agg["min_eval_loss"] < SMOKE_MAX_MIN_EVAL_LOSS, (
        f"min(eval_loss) = {agg['min_eval_loss']:.4f} ≥ "
        f"{SMOKE_MAX_MIN_EVAL_LOSS:.4f}. {cx}"
    )
    assert agg["min_gap"] <= SMOKE_MAX_MIN_GAP, (
        f"min(eval_loss − train_loss) = {agg['min_gap']:.4f} > "
        f"{SMOKE_MAX_MIN_GAP:.4f}. {cx}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 1 — full variant (200-epoch buggy-config training)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_property1_full():
    """Property 1 full — 200-epoch run, asserts full Property-1 thresholds.

    EXPECTED ON UNFIXED CODE: FAILS (matches the published `train_log.txt`
    where eval_exact_match is 0.000 across all 200 epochs and final
    eval_loss = 3.823 > log(25)).

    EXPECTED ON FIXED CODE: PASSES.

    Validates: Requirements 2.1, 2.2, 2.3
    """
    _skip_if_not_hardware_target()
    _require_dataset()

    metrics = _run_buggy_training(epochs=200)
    agg = _aggregate(metrics)

    cx = (
        f"Full run on data/metadata.csv (seed={BUGGY_SEED}, "
        f"{BUGGY_CONFIG}) for 200 epochs: "
        f"max_eval_exact_match={agg['max_eval_exact_match']:.4f}, "
        f"min_eval_loss={agg['min_eval_loss']:.4f}, "
        f"min(eval-train) gap={agg['min_gap']:.4f}, "
        f"final_eval_loss={agg['final_eval_loss']:.4f}, "
        f"final_train_loss={agg['final_train_loss']:.4f}, "
        f"log(NUM_CLASSES)={LOG_NUM_CLASSES:.4f}."
    )

    assert agg["max_eval_exact_match"] >= FULL_MIN_EXACT_MATCH, (
        f"max(eval_exact_match) = {agg['max_eval_exact_match']:.4f} < "
        f"{FULL_MIN_EXACT_MATCH:.4f}. {cx}"
    )
    assert agg["min_eval_loss"] < FULL_MAX_MIN_EVAL_LOSS_VS_LOG_K, (
        f"min(eval_loss) = {agg['min_eval_loss']:.4f} ≥ "
        f"log(NUM_CLASSES) = {FULL_MAX_MIN_EVAL_LOSS_VS_LOG_K:.4f}. {cx}"
    )
    assert agg["min_eval_loss"] < FULL_MAX_MIN_EVAL_LOSS_VS_BUGGY_FLOOR, (
        f"min(eval_loss) = {agg['min_eval_loss']:.4f} ≥ "
        f"{FULL_MAX_MIN_EVAL_LOSS_VS_BUGGY_FLOOR:.4f} (buggy run floor). {cx}"
    )
    assert agg["min_gap"] <= FULL_MAX_MIN_GAP, (
        f"min(eval_loss − train_loss) = {agg['min_gap']:.4f} > "
        f"{FULL_MAX_MIN_GAP:.4f}. {cx}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 1 — GPU utilisation sub-test
# ─────────────────────────────────────────────────────────────────────────────

def _sample_gpu_util_window(
    duration_s: int,
    period_ms: int,
    stop_event: threading.Event,
) -> list[float]:
    """Sample GPU utilization via `nvidia-smi` for `duration_s` seconds.

    Each sample is a percent (0..100). Returns the list of sampled values.
    Stops early if `stop_event` is set.
    """
    samples: list[float] = []
    deadline = time.monotonic() + duration_s
    period_s = period_ms / 1000.0
    while time.monotonic() < deadline and not stop_event.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                timeout=2.0,
            )
            line = out.decode("utf-8", errors="replace").strip().splitlines()[0]
            samples.append(float(line.strip()))
        except (subprocess.SubprocessError, ValueError, IndexError):
            # Silently skip transient nvidia-smi failures; keep sampling.
            pass
        time.sleep(period_s)
    return samples


def test_property1_gpu_util():
    """Property 1 GPU util — assert mean GPU util ≥ 80% in a 30 s window.

    The window is taken mid-training, after the first epoch (warm-up) and
    before the final ONNX export step, so it reflects steady-state.

    EXPECTED ON UNFIXED CODE: FAILS — `num_workers=0` keeps the pipeline
    CPU-bound and GPU util sits around 50%.

    EXPECTED ON FIXED CODE: PASSES.

    Validates: Requirement 2.4
    """
    _skip_if_not_hardware_target(require_smi=True)
    _require_dataset()

    # Run a smoke training on a background thread so we can sample GPU
    # util concurrently from the main thread.
    stop_event = threading.Event()
    samples_holder: dict[str, Any] = {"samples": [], "error": None}

    def _sampler():
        # Skip epoch 0 (warm-up): wait long enough for at least one epoch
        # to start before sampling. With ~10 it/s and 20 batches/epoch,
        # one full epoch ≈ 2 s; we wait 5 s for safety on slower hosts.
        time.sleep(5.0)
        try:
            samples_holder["samples"] = _sample_gpu_util_window(
                duration_s=GPU_UTIL_WINDOW_SECONDS,
                period_ms=GPU_UTIL_SAMPLE_PERIOD_MS,
                stop_event=stop_event,
            )
        except Exception as exc:  # pragma: no cover — diagnostic only
            samples_holder["error"] = repr(exc)

    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()
    try:
        # Use a short smoke run so the test completes in reasonable time.
        # Even 10 epochs is enough to cover the 30 s sampling window plus
        # the warm-up skip.
        _run_buggy_training(epochs=20)
    finally:
        stop_event.set()
        sampler.join(timeout=5.0)

    if samples_holder["error"]:
        pytest.fail(
            f"GPU util sampler errored: {samples_holder['error']}"
        )

    samples = samples_holder["samples"]
    if not samples:
        pytest.skip(
            "No GPU util samples were captured (training likely finished "
            "before the sampler started). Re-run on hardware target or "
            "increase the smoke epoch count."
        )

    mean_util_pct = sum(samples) / len(samples)
    mean_util = mean_util_pct / 100.0

    cx = (
        f"GPU util window: {len(samples)} samples over "
        f"~{GPU_UTIL_WINDOW_SECONDS}s, mean={mean_util:.3f} "
        f"(={mean_util_pct:.1f}%), min={min(samples):.1f}%, "
        f"max={max(samples):.1f}%."
    )

    assert mean_util >= GPU_UTIL_MIN_MEAN, (
        f"mean GPU utilization = {mean_util:.3f} < "
        f"{GPU_UTIL_MIN_MEAN:.3f}. DataLoader is the bottleneck. {cx}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper assertion #1 — Replay the historical training log
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_log_confirms_bug():
    """Helper — parse `train_log.txt` from the historical buggy run.

    Asserts:
        max(eval_exact_match) == 0.000
        min(eval_loss) ≥ 3.388     (the documented epoch-18 floor)
        final eval_loss > log(25)  (val worse than uniform-blank)

    PASSES on UNFIXED code (i.e. on the historical log), confirming the
    bug. After the fix, the log file is replaced with a fresh post-fix
    run and this helper is expected to FAIL — that is the inversion
    exercised by sub-task 3.5.

    Validates: Requirements 2.1, 2.2 (historical evidence of bug)
    """
    log_path = PROJECT_ROOT / "train_log.txt"
    if not log_path.exists():
        pytest.skip(f"train_log.txt missing: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = _parse_metrics_from_output(text)

    eval_em = metrics["eval_exact_match"]
    eval_loss = metrics["eval_loss"]

    assert eval_em, (
        f"No eval_exact_match values parsed out of {log_path}. "
        f"Parse regex may need updating."
    )
    assert eval_loss, (
        f"No eval_loss values parsed out of {log_path}."
    )

    max_em = max(eval_em)
    min_eval_loss = min(eval_loss)
    final_eval_loss = eval_loss[-1]

    cx = (
        f"Replayed {len(eval_em)} epochs from train_log.txt: "
        f"max(eval_exact_match)={max_em:.4f}, "
        f"min(eval_loss)={min_eval_loss:.4f}, "
        f"final(eval_loss)={final_eval_loss:.4f}, "
        f"log(25)={LOG_NUM_CLASSES:.4f}."
    )

    assert max_em == pytest.approx(0.000, abs=1e-6), (
        f"max(eval_exact_match) = {max_em:.4f} ≠ 0.000. Log no longer "
        f"matches the historical buggy run. {cx}"
    )
    assert min_eval_loss >= 3.388, (
        f"min(eval_loss) = {min_eval_loss:.4f} < 3.388. Log no longer "
        f"matches the historical buggy run. {cx}"
    )
    assert final_eval_loss > LOG_NUM_CLASSES, (
        f"final(eval_loss) = {final_eval_loss:.4f} ≤ log(25) = "
        f"{LOG_NUM_CLASSES:.4f}. Log no longer matches the historical "
        f"buggy run. {cx}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper assertion #2 — Doc-drift grep over the 6 doc files
# ─────────────────────────────────────────────────────────────────────────────

def test_doc_drift_confirms_bug():
    """Helper — grep the 6 doc files for `hidden=256` / `Epochs=50`.

    Asserts ≥ 1 match exists across the doc set. PASSES on UNFIXED
    docs (the drift tracked in design §"Hypothesized Root Cause" #4).
    After fix D the same grep returns empty and this test will FAIL —
    that inversion is exercised by sub-task 3.5.

    Validates: Requirement 2.5
    """
    matches: list[tuple[str, int, str]] = []
    missing_docs: list[str] = []

    for rel in DOCS_TO_CHECK_FOR_DRIFT:
        path = PROJECT_ROOT / rel
        if not path.exists():
            missing_docs.append(rel)
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if DOC_DRIFT_PATTERN.search(line):
                matches.append((rel, lineno, line.strip()))

    if missing_docs:
        # We still want the test to confirm drift on whatever subset is
        # present; only fail if NONE of the listed docs exist.
        assert len(missing_docs) < len(DOCS_TO_CHECK_FOR_DRIFT), (
            f"All target docs are missing: {missing_docs}"
        )

    assert matches, (
        "No doc-drift matches found across "
        f"{[d for d in DOCS_TO_CHECK_FOR_DRIFT if (PROJECT_ROOT / d).exists()]}. "
        "Either drift has been fixed (good — re-enable this assertion's "
        "inversion in sub-task 3.5) or the search pattern is too strict."
    )

    # Surface the counterexample in the test output for documentation.
    summary = "\n".join(
        f"  {rel}:{lineno}: {line}" for rel, lineno, line in matches[:10]
    )
    print(
        f"\n[test_doc_drift_confirms_bug] {len(matches)} drift matches "
        f"across {len({m[0] for m in matches})} files:\n{summary}"
    )
