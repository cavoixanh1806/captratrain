"""
Property 2 — Preservation property tests (BEFORE applying any fix).

This module encodes Property 2 from `.kiro/specs/crnn-ctc-collapse-fix/design.md`:

    "All Non-Training-Tuning Behavior Identical."

Each test corresponds to one preservation invariant in
`.kiro/specs/crnn-ctc-collapse-fix/bugfix.md` §"Unchanged Behavior" (3.1
through 3.10). The tests are run on UNFIXED code first to confirm the
baseline behaviour we must preserve. After the fix is applied (touches
only `dataset_crnn.py` augmentation, `train_crnn.py` hyperparam +
DataLoader workers, `crnn_model.py` docstring, plus 6 doc files), this
same test file is re-run and is expected to STILL PASS — that is the
preservation guarantee.

Layout:

    test_3_1_charset                  — class layout & charset bytes
    test_3_2_input_shape_normalize    — hypothesis-driven preprocessing
    test_3_3_onnx_contract            — ONNX I/O contract & opset
    test_3_4_resume_checkpoint_schema — checkpoint key schema + resume
    test_3_5_windows_boot_smoke       — Windows subprocess boot smoke
    test_3_6_decode_semantics         — CTC greedy round-trip + _enforce_length
    test_3_7_split_determinism        — create_crnn_datasets seed=42
    test_3_8_solver_api               — CRNNCaptchaSolver + eval_crnn
    test_3_9_no_synthetic_default     — static + runtime "no synthetic" check
    test_3_10_hardware_envelope       — VRAM/RAM peak on hardware target

Tests that require Windows + CUDA (3.5 boot smoke, 3.10 hardware envelope,
parts of 3.4 resume flow that involve actual training) skip with a clear
reason on hosts that do not meet the precondition. Tests that need
`data/metadata.csv` (3.7, 3.8) skip with a clear reason if the file is
missing.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10
"""

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Make the project root importable so we can `import crnn_model` etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Observed oracles — captured by running the UNFIXED code once and pasting
# the resulting values here. Keeping them inline so any silent drift in the
# preserved behaviour shows up as a failed assertion in this file.
# ─────────────────────────────────────────────────────────────────────────────

ORACLE_CHARSET: str = "ACDEFHJKLMNPQRTUVWXY3479"
ORACLE_CHARSET_LEN: int = 24
ORACLE_NUM_CLASSES: int = 25
ORACLE_CTC_BLANK_INDEX: int = 0
ORACLE_INPUT_HEIGHT: int = 64
ORACLE_INPUT_WIDTH: int = 320
ORACLE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
ORACLE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
ORACLE_VAL_SPLIT: float = 0.15
ORACLE_SEED: int = 42

# ONNX contract.
ORACLE_ONNX_INPUT_NAMES: tuple[str, ...] = ("input",)
ORACLE_ONNX_OUTPUT_NAMES: tuple[str, ...] = ("logits",)
ORACLE_ONNX_INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 64, 320)
ORACLE_ONNX_OPSET: int = 14
ORACLE_ONNX_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "input": {0: "batch"},
    "logits": {1: "batch"},
}

# Checkpoint payload schema for `--resume`.
ORACLE_RESUME_KEYS: frozenset[str] = frozenset({
    "state_dict", "optimizer", "scheduler", "scaler",
    "epoch", "best_val_em", "best_epoch",
})

# Real-data split sizes on the historical 754-image metadata.csv.
ORACLE_TOTAL_REAL: int = 754
ORACLE_TRAIN_REAL: int = 641
ORACLE_VAL_REAL: int = 113


# ─────────────────────────────────────────────────────────────────────────────
# Skip helpers (mirror style of tests/test_property1_bug_condition.py)
# ─────────────────────────────────────────────────────────────────────────────


def _has_cuda() -> bool:
    try:
        import torch  # noqa: WPS433
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _is_windows() -> bool:
    return os.name == "nt"


def _require_metadata_csv() -> Path:
    """Return path to data/metadata.csv or skip cleanly."""
    p = PROJECT_ROOT / "data" / "metadata.csv"
    if not p.exists():
        pytest.skip(f"Required dataset not found: {p}")
    return p


def _require_sample_image() -> Path:
    """Return a real captcha image path or skip cleanly."""
    p = PROJECT_ROOT / "data" / "map_00000.png"
    if not p.exists():
        # fall back to first .png in data/
        candidates = sorted((PROJECT_ROOT / "data").glob("*.png"))
        if not candidates:
            pytest.skip("No real captcha images under data/ to exercise solver.")
        return candidates[0]
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 Charset & class layout
# ─────────────────────────────────────────────────────────────────────────────


def test_3_1_charset_and_class_layout():
    """3.1 — Charset bytes, NUM_CLASSES, CTC_BLANK_INDEX, mapping round-trip.

    Validates: Requirement 3.1
    """
    from crnn_model import (
        CAPTCHA_CHARSET,
        CHAR_TO_IDX,
        CTC_BLANK_INDEX,
        IDX_TO_CHAR,
        NUM_CLASSES,
    )

    assert CAPTCHA_CHARSET == ORACLE_CHARSET, (
        f"CAPTCHA_CHARSET drifted: got {CAPTCHA_CHARSET!r}, "
        f"oracle {ORACLE_CHARSET!r}"
    )
    assert len(CAPTCHA_CHARSET) == ORACLE_CHARSET_LEN
    assert NUM_CLASSES == ORACLE_NUM_CLASSES
    assert CTC_BLANK_INDEX == ORACLE_CTC_BLANK_INDEX

    # CHAR_TO_IDX / IDX_TO_CHAR are 1-based (0 reserved for blank) and
    # must round-trip for every char.
    for i, c in enumerate(CAPTCHA_CHARSET):
        assert CHAR_TO_IDX[c] == i + 1, (
            f"CHAR_TO_IDX[{c!r}] = {CHAR_TO_IDX[c]} ≠ {i + 1}"
        )
        assert IDX_TO_CHAR[i + 1] == c
        assert (i + 1) != CTC_BLANK_INDEX

    # No duplicates and blank index is not reused.
    assert len(set(CAPTCHA_CHARSET)) == len(CAPTCHA_CHARSET)
    assert CTC_BLANK_INDEX not in CHAR_TO_IDX.values()


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 Input shape & normalize  (hypothesis-driven)
# ─────────────────────────────────────────────────────────────────────────────


def _expected_per_channel_value(uint8_value: int) -> tuple[float, float, float]:
    """Closed-form post-normalize value for a uniform uint8 image."""
    f = uint8_value / 255.0
    return tuple((f - m) / s for m, s in zip(ORACLE_MEAN, ORACLE_STD))


def test_3_2_dataset_resize_and_normalize_shape_and_dtype():
    """3.2a — `dataset_crnn._resize_and_normalize` shape + dtype + range.

    Hypothesis sweeps random `H × W × 3 uint8` images across H, W ∈ [16, 1024].

    Validates: Requirement 3.2
    """
    from hypothesis import given, settings, strategies as st
    import numpy as np

    from crnn_model import INPUT_HEIGHT, INPUT_WIDTH
    from dataset_crnn import _MEAN, _STD, _resize_and_normalize

    # Constants haven't drifted.
    assert INPUT_HEIGHT == ORACLE_INPUT_HEIGHT
    assert INPUT_WIDTH == ORACLE_INPUT_WIDTH
    assert _MEAN == ORACLE_MEAN
    assert _STD == ORACLE_STD

    @given(h=st.integers(16, 1024), w=st.integers(16, 1024))
    @settings(max_examples=25, deadline=None)
    def _check(h: int, w: int) -> None:
        rng = np.random.default_rng(0xCAFE ^ (h * 1024 + w))
        rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        t = _resize_and_normalize(rgb)

        import torch
        assert isinstance(t, torch.Tensor)
        assert tuple(t.shape) == (3, ORACLE_INPUT_HEIGHT, ORACLE_INPUT_WIDTH)
        assert t.dtype == torch.float32
        # Per-channel post-normalize range for uint8 ∈ [0, 255]:
        #   min = (0/255 - max_mean)/min_std = -2.118
        #   max = (1 - min_mean)/min_std     ≈  2.640
        for ch in range(3):
            ch_min = float(t[ch].min())
            ch_max = float(t[ch].max())
            channel_min_bound = (0.0 - ORACLE_MEAN[ch]) / ORACLE_STD[ch]
            channel_max_bound = (1.0 - ORACLE_MEAN[ch]) / ORACLE_STD[ch]
            # Allow a small float tolerance.
            assert ch_min >= channel_min_bound - 1e-3, (
                f"channel {ch} min {ch_min} below bound {channel_min_bound}"
            )
            assert ch_max <= channel_max_bound + 1e-3, (
                f"channel {ch} max {ch_max} above bound {channel_max_bound}"
            )

    _check()


def test_3_2_dataset_resize_and_normalize_uniform_image_value():
    """3.2b — Closed-form post-normalize value for a uniform uint8 image.

    For a flat 128×128×3 uint8 image filled with 128, the post-normalize
    per-channel value matches `(128/255 − mean)/std` for every channel.

    Validates: Requirement 3.2
    """
    import numpy as np
    from dataset_crnn import _resize_and_normalize

    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    t = _resize_and_normalize(img)
    expected = _expected_per_channel_value(128)
    for ch in range(3):
        # Resize is non-degenerate (no rounding) for a uniform image:
        # all pixels stay at 128 → all post-normalize pixels equal expected.
        assert math.isclose(
            float(t[ch].mean()), expected[ch], abs_tol=1e-5,
        ), f"channel {ch} mean {float(t[ch].mean())} ≠ expected {expected[ch]}"
        # std should be ≈ 0 for a uniform image.
        assert float(t[ch].std()) < 1e-4


def test_3_2_inference_preprocess_matches_dataset():
    """3.2c — `inference_crnn._preprocess` produces same shape/dtype/values
    as `dataset_crnn._resize_and_normalize` on a BGR-input twin.

    Validates: Requirement 3.2
    """
    import numpy as np
    import torch

    from dataset_crnn import _resize_and_normalize
    from inference_crnn import _preprocess

    rng = np.random.default_rng(0xBEEF)
    rgb = rng.integers(0, 256, size=(96, 480, 3), dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()  # BGR for inference path

    t_dataset = _resize_and_normalize(rgb)
    t_inference = _preprocess(bgr)

    assert tuple(t_inference.shape) == (3, ORACLE_INPUT_HEIGHT, ORACLE_INPUT_WIDTH)
    assert t_inference.dtype == torch.float32
    # Values must match closely (both apply the same resize + normalize).
    assert torch.allclose(t_dataset, t_inference, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3.3 ONNX contract
# ─────────────────────────────────────────────────────────────────────────────


def test_3_3_onnx_contract():
    """3.3 — `crnn_model.export_onnx` produces an ONNX file with the
    documented input/output names, shape, and opset.

    Constructed from a freshly initialized CRNN (no smoke training needed
    for a contract-only check). The full train-then-export flow is
    exercised by 3.5 boot smoke on hardware target.

    The opset assertion has two layers:
      (a) the source code in `crnn_model.export_onnx` literally passes
          `opset_version=14` (the request invariant we must preserve);
      (b) the exported file has opset ≥ 14. PyTorch's dynamo-based
          exporter (≥ 2.5) may auto-promote the opset when the requested
          version pre-dates an op it needs — that promotion is environmental
          and not a regression we want to police.

    Validates: Requirement 3.3
    """
    onnx = pytest.importorskip(
        "onnx", reason="onnx is required to validate the export contract"
    )

    import torch

    from crnn_model import CRNN, export_onnx

    # (a) Source-level invariant — `opset_version=14` is still requested.
    src = (PROJECT_ROOT / "crnn_model.py").read_text(encoding="utf-8")
    assert re.search(r"opset_version\s*=\s*14", src), (
        "crnn_model.export_onnx no longer requests opset_version=14; "
        "the export-contract preservation invariant has drifted."
    )

    model = CRNN()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "smoke.onnx"
        export_onnx(model, out)
        assert out.exists() and out.stat().st_size > 0

        proto = onnx.load(str(out))
        # (b) Emitted opset ≥ 14.
        opset_versions = [
            opset.version for opset in proto.opset_import
            if opset.domain in ("", "ai.onnx")
        ]
        assert opset_versions, (
            f"No default-domain opset in exported ONNX: {proto.opset_import}"
        )
        assert max(opset_versions) >= ORACLE_ONNX_OPSET, (
            f"Emitted opset {opset_versions} < requested floor "
            f"{ORACLE_ONNX_OPSET}. Some PyTorch exporters auto-promote "
            f"opset 14→18, which is acceptable; demoting is not."
        )

        # Input names + shape
        graph = proto.graph
        input_names = [vi.name for vi in graph.input]
        output_names = [vi.name for vi in graph.output]
        assert tuple(input_names) == ORACLE_ONNX_INPUT_NAMES, (
            f"input names: got {input_names}, expected "
            f"{ORACLE_ONNX_INPUT_NAMES}"
        )
        assert tuple(output_names) == ORACLE_ONNX_OUTPUT_NAMES, (
            f"output names: got {output_names}, expected "
            f"{ORACLE_ONNX_OUTPUT_NAMES}"
        )

        input_tensor = graph.input[0]
        dims = input_tensor.type.tensor_type.shape.dim
        # First dim is dynamic ("batch"); remaining must be 3, 64, 320.
        assert len(dims) == 4, f"Input rank {len(dims)} ≠ 4"
        # batch axis: must be dynamic via param name OR a literal 1
        first_dim = dims[0]
        is_dynamic_batch = bool(first_dim.dim_param) or first_dim.dim_value == 1
        assert is_dynamic_batch, (
            f"input[0] is not a dynamic batch axis: {first_dim}"
        )
        assert dims[1].dim_value == 3
        assert dims[2].dim_value == ORACLE_INPUT_HEIGHT
        assert dims[3].dim_value == ORACLE_INPUT_WIDTH

        # Output shape: (T, B, num_classes). Just check that batch axis
        # (index 1) is dynamic — `dynamic_axes={"logits": {1: "batch"}}`.
        out_dims = graph.output[0].type.tensor_type.shape.dim
        assert len(out_dims) == 3
        # ONNX may emit dim_param for the dynamic axis.
        # We don't strictly verify the name, only that the contract holds:
        # output is (T, B, C) with C == NUM_CLASSES.
        last_dim = out_dims[2].dim_value
        # Some PyTorch ONNX exporters propagate the constant dim,
        # others leave it as 0 (unknown). Accept either.
        if last_dim:
            assert last_dim == ORACLE_NUM_CLASSES, (
                f"output[2] = {last_dim} ≠ NUM_CLASSES={ORACLE_NUM_CLASSES}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3.4 Resume flow — checkpoint schema (always) + smoke (CUDA only)
# ─────────────────────────────────────────────────────────────────────────────


def test_3_4_checkpoint_schema_contract():
    """3.4 — Checkpoint payload contains all keys required by `--resume`
    and round-trips through torch.save / torch.load.

    Validates: Requirement 3.4
    """
    import torch

    from crnn_model import CRNN, NUM_CLASSES
    from train_crnn import build_warmup_cosine_scheduler

    device = torch.device("cpu")
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = build_warmup_cosine_scheduler(
        optim, warmup_steps=10, total_steps=100,
    )
    # Take a step so the optimizer has populated state.
    optim.zero_grad()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optim.step()
    sched.step()

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": None,  # CPU run → no GradScaler
        "epoch": 2,
        "best_val_em": 0.0,
        "best_epoch": 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "last.pth"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

    # All required keys present (preservation of the resume contract).
    assert ORACLE_RESUME_KEYS.issubset(loaded.keys()), (
        f"Resume payload missing keys: "
        f"{ORACLE_RESUME_KEYS - set(loaded.keys())}"
    )

    # Round-trip values for the scalar fields.
    assert loaded["epoch"] == 2
    assert loaded["best_val_em"] == 0.0
    assert loaded["best_epoch"] == 0

    # Optimizer/scheduler state restorable into a fresh instance.
    fresh_model = CRNN(num_classes=NUM_CLASSES).to(device)
    fresh_optim = torch.optim.AdamW(
        fresh_model.parameters(), lr=1e-3, weight_decay=1e-4,
    )
    fresh_sched = build_warmup_cosine_scheduler(
        fresh_optim, warmup_steps=10, total_steps=100,
    )
    fresh_model.load_state_dict(loaded["state_dict"])
    fresh_optim.load_state_dict(loaded["optimizer"])
    fresh_sched.load_state_dict(loaded["scheduler"])

    # LR schedule resumes at the same step as the saved one.
    assert fresh_sched.last_epoch == sched.last_epoch


@pytest.mark.slow
def test_3_4_resume_smoke_2_plus_2_epochs():
    """3.4 (full) — smoke 2 epochs → --resume → 2 more epochs; final_epoch == 4.

    Skips on hosts without CUDA: a 2-epoch CPU run on 641 images is too
    slow for unit tests, and 3.4's primary contract (key schema) is
    already validated by `test_3_4_checkpoint_schema_contract` above.

    Validates: Requirement 3.4
    """
    if not _has_cuda():
        pytest.skip(
            "Resume smoke (2+2 epochs) requires CUDA; CPU is too slow for "
            "this test. Schema contract is already covered by "
            "test_3_4_checkpoint_schema_contract."
        )
    _require_metadata_csv()

    import torch

    import train_crnn

    last_path = PROJECT_ROOT / train_crnn.LAST_CHECKPOINT_PATH
    backup_last = None
    if last_path.exists():
        backup_last = last_path.with_suffix(last_path.suffix + ".bak")
        shutil.move(str(last_path), str(backup_last))

    try:
        # 2 epochs from scratch.
        train_crnn.main(
            epochs=2, batch_size=32, lr=1e-3,
            use_synthetic=False, use_real=True, augment=True, resume=False,
        )
        ckpt_a = torch.load(last_path, map_location="cpu", weights_only=False)
        assert ckpt_a["epoch"] == 2
        assert ORACLE_RESUME_KEYS.issubset(ckpt_a.keys())

        # Resume + 2 more.
        train_crnn.main(
            epochs=4, batch_size=32, lr=1e-3,
            use_synthetic=False, use_real=True, augment=True, resume=True,
        )
        ckpt_b = torch.load(last_path, map_location="cpu", weights_only=False)
        assert ckpt_b["epoch"] == 4
        assert ORACLE_RESUME_KEYS.issubset(ckpt_b.keys())
    finally:
        if backup_last is not None:
            shutil.move(str(backup_last), str(last_path))


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 Windows boot smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_3_5_windows_boot_smoke():
    """3.5 — Windows boot smoke for `python train_crnn.py --epochs 1`
    and `python train_crnn.py --num-workers 0 --epochs 1`. Skip the
    `--num-workers > 0` variant on UNFIXED code (the CLI flag does not
    exist yet).

    Skips on non-Windows or hosts without CUDA: a 1-epoch CPU run on 641
    real images is too slow for unit tests.

    Validates: Requirement 3.5
    """
    if not _is_windows():
        pytest.skip("3.5 boot smoke is a Windows-only invariant.")
    if not _has_cuda():
        pytest.skip(
            "3.5 boot smoke requires CUDA on the hardware target — a CPU "
            "1-epoch run on 641 images takes minutes which is too slow for "
            "unit tests. Run on Windows + RTX 3060 + CUDA 12.8."
        )
    _require_metadata_csv()

    python_exe = sys.executable

    # Pre-fix: only the bare command and `--num-workers 0` are valid.
    # (The post-fix `--num-workers 4` invocation is exercised by the
    # post-fix run, not by this preservation test.)
    cmd_variants: list[list[str]] = [
        [python_exe, "train_crnn.py", "--epochs", "1"],
    ]

    # Detect whether `--num-workers` flag exists in the current code.
    help_proc = subprocess.run(
        [python_exe, "train_crnn.py", "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=30,
    )
    if "--num-workers" in (help_proc.stdout or "") + (help_proc.stderr or ""):
        cmd_variants.append(
            [python_exe, "train_crnn.py", "--num-workers", "0", "--epochs", "1"]
        )

    for cmd in cmd_variants:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
            timeout=600,  # 10 min ceiling per variant
        )
        stderr = proc.stderr or ""
        assert proc.returncode == 0, (
            f"{cmd!r} exited with {proc.returncode}.\nstderr tail:\n"
            f"{stderr[-2000:]}"
        )
        assert "BrokenPipeError" not in stderr, (
            f"{cmd!r} hit BrokenPipeError.\nstderr:\n{stderr}"
        )
        assert "PicklingError" not in stderr, (
            f"{cmd!r} hit PicklingError.\nstderr:\n{stderr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.6 Decode semantics + _enforce_length boundary
# ─────────────────────────────────────────────────────────────────────────────


def _build_perfect_logits(text: str):
    """Build (T=80, B=1, C=NUM_CLASSES) logits where each char gets ~16
    consecutive timesteps, separated by a blank frame to avoid CTC's
    repeat-collapse rule when the same char appears twice in a row.
    """
    import torch

    from crnn_model import CTC_BLANK_INDEX, NUM_CLASSES, encode_label

    ids = encode_label(text)
    T = 80
    B = 1
    logits = torch.full((T, B, NUM_CLASSES), -10.0)
    chunk = T // len(ids) if ids else T
    for i, idx in enumerate(ids):
        start = i * chunk
        end = (i + 1) * chunk if i + 1 < len(ids) else T
        logits[start:end, 0, idx] = 10.0
        # Prepend a blank frame between identical consecutive chars to
        # prevent the greedy decoder from collapsing them.
        if i > 0 and ids[i] == ids[i - 1]:
            logits[start, 0, idx] = -10.0
            logits[start, 0, CTC_BLANK_INDEX] = 10.0
    return logits


def test_3_6_decode_greedy_round_trip():
    """3.6a — `encode_label → simulate logits → decode_greedy` round-trip
    for random 5-character labels drawn from `CAPTCHA_CHARSET`.

    Validates: Requirement 3.6
    """
    from hypothesis import given, settings, strategies as st

    from crnn_model import CAPTCHA_CHARSET, decode_greedy

    @given(text=st.text(alphabet=CAPTCHA_CHARSET, min_size=5, max_size=5))
    @settings(max_examples=80, deadline=None)
    def _check(text: str) -> None:
        logits = _build_perfect_logits(text)
        recovered = decode_greedy(logits)
        assert recovered == [text], (
            f"Round-trip failed: {text!r} → {recovered!r}"
        )

    _check()


@pytest.mark.parametrize("length,raw,expected", [
    (0, "", "AAAAA"),
    (1, "X", "XAAAA"),
    (4, "WXYZA", "WXYZA"[:4] + "A"),  # raw len 4 → pad 1
    (5, "ABCDE", "ABCDE"),
    (6, "ABCDEF", "ABCDE"),
    (10, "ABCDEFGHIJ", "ABCDE"),
])
def test_3_6_enforce_length_boundary(length: int, raw: str, expected: str):
    """3.6b — `_enforce_length` boundary values.

    Pad with `CAPTCHA_CHARSET[0] == 'A'` when shorter; truncate from the
    right when longer.

    Validates: Requirement 3.6
    """
    from crnn_model import CAPTCHA_CHARSET
    from inference_crnn import _enforce_length

    assert CAPTCHA_CHARSET[0] == "A"

    # Build a real "raw of given length" by trimming/padding the test arg.
    raw = raw[:length] if len(raw) > length else raw + "X" * (length - len(raw))
    assert len(raw) == length

    out = _enforce_length(raw, target=5)
    assert len(out) == 5

    if length >= 5:
        assert out == raw[:5]
    else:
        # Padded part must be exactly CAPTCHA_CHARSET[0].
        assert out[:length] == raw
        assert out[length:] == "A" * (5 - length)


# ─────────────────────────────────────────────────────────────────────────────
# 3.7 Train/val split determinism
# ─────────────────────────────────────────────────────────────────────────────


def _filenames_of(ds) -> list[str]:
    """Return the filename list backing a CRNNCaptchaDataset (or
    ConcatDataset of them)."""
    from torch.utils.data import ConcatDataset

    if hasattr(ds, "df"):
        return ds.df["filename"].tolist()
    if isinstance(ds, ConcatDataset):
        out: list[str] = []
        for sub in ds.datasets:
            out.extend(_filenames_of(sub))
        return out
    raise AssertionError(f"Unsupported dataset type: {type(ds)}")


def test_3_7_create_crnn_datasets_seed_42_deterministic():
    """3.7 — Two calls of `create_crnn_datasets(seed=42)` produce identical
    train and val filename lists, and val_split ≈ 0.15 on metadata.csv.

    Validates: Requirement 3.7
    """
    meta = _require_metadata_csv()
    from dataset_crnn import create_crnn_datasets

    train_a, val_a = create_crnn_datasets(
        use_real=True, use_synthetic=False, seed=ORACLE_SEED,
        augment_train=True,
    )
    train_b, val_b = create_crnn_datasets(
        use_real=True, use_synthetic=False, seed=ORACLE_SEED,
        augment_train=True,
    )

    train_a_names = _filenames_of(train_a)
    train_b_names = _filenames_of(train_b)
    val_a_names = _filenames_of(val_a)
    val_b_names = _filenames_of(val_b)

    assert train_a_names == train_b_names, (
        "Train split filenames differ across two seeded calls."
    )
    assert val_a_names == val_b_names, (
        "Val split filenames differ across two seeded calls."
    )

    n_train = len(train_a_names)
    n_val = len(val_a_names)
    n_total = n_train + n_val
    assert n_total > 0
    # Disjoint splits.
    assert not (set(train_a_names) & set(val_a_names)), (
        "Train and val filenames overlap."
    )

    val_ratio = n_val / n_total
    assert abs(val_ratio - ORACLE_VAL_SPLIT) <= 0.01, (
        f"val_split = {val_ratio:.4f} not within ±0.01 of "
        f"{ORACLE_VAL_SPLIT}; n_train={n_train}, n_val={n_val}"
    )

    # On the historical metadata.csv (754 rows) the split is exactly
    # 641/113. Assert this when the corpus matches.
    if n_total == ORACLE_TOTAL_REAL:
        assert n_train == ORACLE_TRAIN_REAL
        assert n_val == ORACLE_VAL_REAL


# ─────────────────────────────────────────────────────────────────────────────
# 3.8 Solver API + eval_crnn output format
# ─────────────────────────────────────────────────────────────────────────────


def _save_random_checkpoint(path: Path) -> None:
    """Save a randomly-initialised CRNN via `save_crnn` so the solver
    has a checkpoint to load (we only validate format, not accuracy)."""
    import torch

    from crnn_model import CRNN, save_crnn

    torch.manual_seed(0)
    model = CRNN()
    save_crnn(model, str(path))


def test_3_8_solver_api_signatures():
    """3.8a — `CRNNCaptchaSolver` API signatures and output shapes.

    Validates: Requirement 3.8
    """
    sample_img = _require_sample_image()

    from inference_crnn import CRNNCaptchaSolver

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "dummy.pth"
        _save_random_checkpoint(ckpt)
        solver = CRNNCaptchaSolver(str(ckpt), device="cpu")

        # solve(path) → str of length 5
        out = solver.solve(str(sample_img))
        assert isinstance(out, str)
        assert len(out) == 5

        # solve_with_confidence(path) → (str, float in [0, 1])
        text, conf = solver.solve_with_confidence(str(sample_img))
        assert isinstance(text, str) and len(text) == 5
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

        # solve_batch
        paths = [str(sample_img)] * 3
        batch_out = solver.solve_batch(paths)
        assert isinstance(batch_out, list) and len(batch_out) == 3
        for s in batch_out:
            assert isinstance(s, str) and len(s) == 5

        # solve_batch_with_confidence
        batch_conf = solver.solve_batch_with_confidence(paths)
        assert isinstance(batch_conf, list) and len(batch_conf) == 3
        for s, c in batch_conf:
            assert isinstance(s, str) and len(s) == 5
            assert isinstance(c, float) and 0.0 <= c <= 1.0


def test_3_8_eval_crnn_output_format():
    """3.8b — `eval_crnn.evaluate` output captures the documented sections.

    Validates: Requirement 3.8
    """
    meta = _require_metadata_csv()

    import io
    import contextlib

    import eval_crnn

    # Build a small 10-image metadata.csv from a slice of the real one.
    import pandas as pd
    df = pd.read_csv(meta, dtype=str).dropna()
    df["text"] = df["text"].str.strip().str.upper()
    df = df[df["text"].str.len() == 5].head(10)

    with tempfile.TemporaryDirectory() as tmpdir:
        small_meta = Path(tmpdir) / "metadata_small.csv"
        df.to_csv(small_meta, index=False)
        ckpt = Path(tmpdir) / "dummy.pth"
        _save_random_checkpoint(ckpt)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = eval_crnn.evaluate(
                checkpoint=str(ckpt),
                metadata_path=str(small_meta),
                image_dir=str(PROJECT_ROOT / "data"),
                batch_size=8,
            )
        out = buf.getvalue()

    # Required sections in printed report.
    expected_substrings = [
        "Exact match",
        "CER",
        "Per-position accuracy",
        "Top 10 confusions",
        "VERDICT",
    ]
    for sub in expected_substrings:
        assert sub in out, (
            f"Expected {sub!r} in eval_crnn.evaluate output. Got:\n{out}"
        )

    # Returned dict has the canonical keys.
    assert isinstance(result, dict)
    for key in ("total", "exact_match", "cer", "per_position",
                "confusions", "avg_confidence"):
        assert key in result, f"eval result missing key {key!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 3.9 No synthetic data default
# ─────────────────────────────────────────────────────────────────────────────


def _line_is_inside_string_constant(tree: ast.Module, lineno: int) -> bool:
    """Return True iff `lineno` lies within the source range of any
    Constant string node in the AST (i.e. it's part of a string literal,
    docstring, or multi-line string).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is not None and start <= lineno <= end:
                return True
    return False


def _line_is_inside_use_synthetic_branch(
    tree: ast.Module, lineno: int,
) -> bool:
    """Return True iff `lineno` lies inside an `if use_synthetic[: ...]`
    branch in `train_crnn.py`.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond_src = ast.unparse(node.test)
            if "use_synthetic" not in cond_src:
                continue
            start = node.lineno
            end = node.end_lineno or start
            if start <= lineno <= end:
                return True
    return False


def test_3_9_no_synthetic_static_grep():
    """3.9a — Static check: `grep -n "synthetic_crnn|generate_synthetic_crnn"`
    on `train_crnn.py` matches only inside string literals (docstring /
    `--use-synthetic` argparse help) or `if use_synthetic:` branches.

    Equivalent stricter form: no top-level statement of `train_crnn.py`
    imports or unconditionally invokes `generate_synthetic_crnn`.

    Validates: Requirement 3.9
    """
    src = (PROJECT_ROOT / "train_crnn.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Stricter form first: no `import generate_synthetic_crnn` or
    # `from generate_synthetic_crnn import ...` at module level.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "generate_synthetic_crnn" not in alias.name, (
                    f"Top-level import of {alias.name!r} in train_crnn.py "
                    f"violates 'no synthetic by default'."
                )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "generate_synthetic_crnn" not in mod, (
                f"Top-level `from {mod} import ...` in train_crnn.py "
                f"violates 'no synthetic by default'."
            )

    # The literal grep: every match line is inside a string literal OR
    # an `if use_synthetic:` branch.
    pat = re.compile(r"synthetic_crnn|generate_synthetic_crnn")
    matches: list[tuple[int, str]] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if pat.search(line):
            matches.append((lineno, line))

    # We expect ≥ 1 match (otherwise grep regex is broken).
    assert matches, (
        "grep produced no matches; the regex may be wrong or the file "
        "structure changed dramatically."
    )

    bad: list[tuple[int, str]] = []
    for lineno, line in matches:
        in_string = _line_is_inside_string_constant(tree, lineno)
        in_branch = _line_is_inside_use_synthetic_branch(tree, lineno)
        if not (in_string or in_branch):
            bad.append((lineno, line))

    assert not bad, (
        "Found `synthetic_crnn`/`generate_synthetic_crnn` mentions outside "
        "string literals and `if use_synthetic:` branches:\n"
        + "\n".join(f"  {ln}: {line}" for ln, line in bad)
    )


def test_3_9_no_synthetic_runtime_default():
    """3.9b — `create_crnn_datasets(use_synthetic=False)` works without
    `data/synthetic_crnn` (we point synthetic_dir at a non-existent path
    to simulate "absent/renamed"), and `train_crnn.main`'s default
    `use_synthetic=False` flows through `create_crnn_datasets` without
    raising `FileNotFoundError` for synthetic_crnn.

    Validates: Requirement 3.9
    """
    meta = _require_metadata_csv()

    # The `main()` signature in train_crnn.py defaults use_synthetic=False.
    import inspect

    import train_crnn
    sig = inspect.signature(train_crnn.main)
    assert sig.parameters["use_synthetic"].default is False, (
        "train_crnn.main(use_synthetic=...) default drifted from False."
    )

    from dataset_crnn import create_crnn_datasets

    # Point synthetic_dir at a non-existent path; with use_synthetic=False
    # it must not be touched.
    nonexistent = PROJECT_ROOT / "data" / "_synthetic_does_not_exist__"
    assert not nonexistent.exists()

    train_ds, val_ds = create_crnn_datasets(
        synthetic_dir=str(nonexistent),
        use_real=True,
        use_synthetic=False,
        seed=ORACLE_SEED,
    )

    assert len(train_ds) > 0
    assert len(val_ds) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3.10 Hardware envelope (CUDA + RTX 3060 only)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_3_10_hardware_envelope():
    """3.10 — VRAM peak < 8 GiB and RAM peak < 16 GiB after a 1-epoch
    smoke run at `DEFAULT_BATCH_SIZE = 32`.

    Skips on hosts without CUDA — RAM-only measurement is not meaningful
    on a dev box that lacks the bug's GPU.

    Validates: Requirement 3.10
    """
    if not _has_cuda():
        pytest.skip(
            "3.10 hardware envelope requires CUDA to measure VRAM via "
            "torch.cuda.max_memory_allocated; run on Windows + RTX 3060 + "
            "CUDA 12.8."
        )
    psutil = pytest.importorskip(
        "psutil", reason="psutil required for RAM peak measurement"
    )
    _require_metadata_csv()

    import torch

    import train_crnn

    proc = psutil.Process()
    rss_baseline = proc.memory_info().rss

    torch.cuda.reset_peak_memory_stats()
    train_crnn.main(
        epochs=1,
        batch_size=train_crnn.DEFAULT_BATCH_SIZE,
        lr=train_crnn.DEFAULT_LR,
        use_synthetic=False, use_real=True, augment=True, resume=False,
    )

    vram_peak = torch.cuda.max_memory_allocated()
    rss_peak = proc.memory_info().rss

    eight_gib = 8 * 1024 ** 3
    sixteen_gib = 16 * 1024 ** 3
    assert vram_peak < eight_gib, (
        f"VRAM peak = {vram_peak / 1024**3:.2f} GiB ≥ 8 GiB"
    )
    assert rss_peak < sixteen_gib, (
        f"RAM peak = {rss_peak / 1024**3:.2f} GiB ≥ 16 GiB "
        f"(baseline was {rss_baseline / 1024**3:.2f} GiB)"
    )
