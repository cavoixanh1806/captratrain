# Implementation Plan

## Overview

Bugfix implementation for `crnn-ctc-collapse-fix` follows the bug-condition methodology from the design document. The plan has three phases:

1. **Explore** — Write a property-based exploration test (Property 1: Bug Condition) on UNFIXED code; the test MUST FAIL on unfixed code, demonstrating CTC collapse + augmentation overfit + GPU under-utilization on the target hardware (Windows + i5-12400F + RTX 3060 8GB + CUDA 12.8).
2. **Preserve** — Write property-based preservation tests (Property 2: Preservation) capturing the 10 unchanged behaviours (3.1 – 3.10) from the design's Preservation Requirements; tests MUST PASS on UNFIXED code, encoding the baseline behaviour we must preserve.
3. **Implement & Validate** — Apply the four-part fix (A: dataset_crnn augmentation tuning, B: train_crnn hyperparam + DataLoader workers, C: crnn_model architecture canonicalization, D: doc/code reconciliation across 6 docs); re-run Property 1 (now PASSES → bug fixed) and Property 2 (still PASSES → no regressions).

The exploration and preservation tests live as standalone tasks BEFORE the implementation parent task, per the bugfix workflow contract. The implementation task carries `_Bug_Condition`, `_Expected_Behavior`, and `_Preservation` annotations referencing the design's pseudocode and specifications.

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 1,
      "tasks": ["1", "2"],
      "description": "Write Property 1 exploration test (fails on unfixed code) and Property 2 preservation tests (pass on unfixed code) in parallel."
    },
    {
      "wave": 2,
      "tasks": ["3.1", "3.2", "3.3", "3.4"],
      "description": "Apply the four independent sub-fixes in parallel: dataset_crnn augmentation, train_crnn hyperparam + DataLoader, crnn_model canonicalization, doc/code reconciliation."
    },
    {
      "wave": 3,
      "tasks": ["3.5", "3.6"],
      "description": "Re-run Property 1 (now PASSES → bug fixed) and Property 2 (still PASSES → no regressions) in parallel; both depend on all four sub-fixes from wave 2."
    },
    {
      "wave": 4,
      "tasks": ["4"],
      "description": "Final checkpoint — full suite green, artefacts spot-checked."
    }
  ]
}
```

```
Task 1 (Property 1: Bug Condition exploration test, fails on unfixed code)
  │
  └──> Task 3 (Implementation - parent)
         │
         ├── 3.1 Fix A — dataset_crnn.py augmentation
         ├── 3.2 Fix B — train_crnn.py hyperparam + DataLoader
         ├── 3.3 Fix C — crnn_model.py canonicalization
         ├── 3.4 Fix D — doc/code reconciliation
         ├── 3.5 Verify Property 1 now passes  (depends on 3.1, 3.2, 3.3, 3.4)
         └── 3.6 Verify Property 2 still passes (depends on 3.1, 3.2, 3.3, 3.4)
                  │
                  └──> Task 4 (Checkpoint)

Task 2 (Property 2: Preservation tests, pass on unfixed code)
  │
  └──> Task 3.6 (re-run on fixed code, must still pass)
         │
         └──> Task 4 (Checkpoint)
```

Order constraints:
- Tasks 1 and 2 are independent of each other but BOTH must complete before Task 3.
- Sub-tasks 3.1 – 3.4 are independent and may be applied in any order; all four must be in place before 3.5 / 3.6.
- Sub-tasks 3.5 and 3.6 must both pass before Task 4.

## Tasks

- [x] 1. Write bug condition exploration test (BEFORE applying any fix)
  - **Property 1: Bug Condition** - Trained CRNN+CTC Generalises To Real Val And GPU Saturated
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists.
  - **DO NOT attempt to fix the test or the code when it fails**.
  - **NOTE**: This test encodes the expected post-fix behavior — it will validate the fix when it passes after implementation.
  - **GOAL**: Surface counterexamples that demonstrate CTC collapse + augmentation overfit + GPU under-utilization on hardware target (Windows + i5-12400F + RTX 3060 8GB + CUDA 12.8).
  - **Scoped PBT Approach** (deterministic bug → scope property to concrete failing case): use the exact buggy config from `isBugCondition` in design — `data/metadata.csv` (754 real images, train=641/val=113), `seed=42`, `_TRAIN_AUG = _build_albu_aug(strong=True)`, `DEFAULT_LR = 1e-3`, `WARMUP_STEPS = 200`, `num_workers = 0`, `batch_size = 32`, hardware target. The "for-all" generalisation is over the (deterministic) re-runs of this single config.
  - Implement as `tests/test_property1_bug_condition.py` using `pytest`. Provide two variants:
    - `test_property1_smoke` — runs `train_crnn.main(epochs=20, …)` on the buggy config; parses metrics from logger output / returned dict; asserts `max(eval_exact_match) ≥ 0.05` AND `min(eval_loss) < 3.0` AND `min(eval_loss − train_loss) ≤ 0.7` (smoke sub-thresholds from design "Integration Tests" §1).
    - `test_property1_full` — runs `train_crnn.main(epochs=200, …)` and asserts the full Property 1 thresholds: `max(eval_exact_match) ≥ 0.30` AND `min(eval_loss) < log(25) ≈ 3.219` AND `min(eval_loss) < 3.388` AND `min(eval_loss − train_loss) ≤ 0.50`. Mark with `@pytest.mark.slow` so it can be opted-in.
    - `test_property1_gpu_util` — while a smoke run is mid-training, sample `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000` for a continuous 30s window (skipping epoch 0, eval step, ONNX export step); assert `mean(util) ≥ 0.80`. On non-Windows / no-GPU hosts, skip with a clear reason.
  - Optional supporting checks (from design "Exploratory Bug Condition Checking") — implement as helper assertions inside the same test module to confirm the hypothesised root cause:
    - Replay parser over existing `train_log.txt`: assert `max(eval_exact_match) == 0.000` AND `min(eval_loss) ≥ 3.388` AND final `eval_loss > log(25)`. PASSES on unfixed log → confirms bug.
    - Doc-drift grep over `{README.md, PIPELINE_SUMMARY.md, CLAUDE.md, docs/adr/0001-crnn-ctc-over-softmax.md, docs/codebase/ARCHITECTURE.md, docs/codebase/CONCERNS.md}` for `hidden(_size)?\s*=?\s*256` or `Epochs\s*=\s*50`: assert ≥ 1 match exists. PASSES on unfixed docs → confirms drift.
  - Run test on UNFIXED code:
    - **EXPECTED OUTCOME**: `test_property1_smoke` and `test_property1_full` FAIL on unfixed code (this is correct — it proves the bug exists). `test_property1_gpu_util` FAILS on unfixed code (mean GPU util < 80%). Replay-log helper PASSES (confirms historical run). Doc-drift helper PASSES (confirms drift).
  - Document counterexamples found, e.g.:
    - "200-epoch run on `data/metadata.csv` with `seed=42` produced `eval_exact_match = 0.000` for all epochs, final `eval_loss = 3.823 > log(25) = 3.219`, gap = +1.25 nats."
    - "GPU util sampled at 30s window during epoch 5 averaged ≈ 50% on RTX 3060 with 12 CPU threads at ~100%."
    - "Doc-drift grep matched 6+ occurrences of `hidden=256` / `Epochs=50` across README/PIPELINE_SUMMARY/CLAUDE/ADR-0001/ARCHITECTURE/CONCERNS."
  - Mark task complete when test is written, run on unfixed code, and failures are documented.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Write preservation property tests (BEFORE applying any fix)
  - **Property 2: Preservation** - All Non-Training-Tuning Behavior Identical
  - **IMPORTANT**: Follow observation-first methodology — observe behaviour on UNFIXED code first, capture as oracle, then encode as property tests.
  - Implement as `tests/test_property2_preservation.py` using `pytest` + `hypothesis` (dev-only dependency). Cover the full Preservation Set from design §"Preservation Requirements" (3.1 – 3.10):
    - **3.1 Charset & class layout** — assert `CAPTCHA_CHARSET == "ACDEFHJKLMNPQRTUVWXY3479"` AND `len(CAPTCHA_CHARSET) == 24` AND `NUM_CLASSES == 25` AND `CTC_BLANK_INDEX == 0` AND `CHAR_TO_IDX`/`IDX_TO_CHAR` round-trip 1-based for every char.
    - **3.2 Input shape & normalize** — `@given(st.integers(16, 1024), st.integers(16, 1024))` synth random `H×W×3 uint8`, run `dataset_crnn._resize_and_normalize`, assert `shape == (3, 64, 320)`, `dtype == float32`, mean/std post-normalize within ImageNet expected range. Same property for `inference_crnn._preprocess`.
    - **3.3 ONNX contract** — after a 1-epoch smoke train, call `crnn_model.export_onnx`; load resulting file via `onnx.load`; assert `input_names == ["input"]`, `output_names == ["logits"]`, input shape `[batch, 3, 64, 320]`, `opset_version == 14`, `dynamic_axes` matches design.
    - **3.4 Resume flow** — smoke 2 epochs → `--resume` → 2 more epochs; assert `final_epoch == 4`, optimizer/scheduler/scaler state restored, keys `{state_dict, optimizer, scheduler, scaler, epoch, best_val_em, best_epoch}` present in checkpoint.
    - **3.5 Windows boot smoke** — on Windows, `python train_crnn.py --epochs 1` AND `python train_crnn.py --num-workers 0 --epochs 1` AND (post-fix) `python train_crnn.py --num-workers 4 --epochs 1` all exit 0 with no `BrokenPipeError`/`PicklingError` in stderr. Skip on non-Windows with reason.
    - **3.6 Decode semantics** — `@given(st.text(alphabet=CAPTCHA_CHARSET, min_size=5, max_size=5))` round-trip `encode_label → simulate perfectly-confident logits → decode_greedy → assert recovered == input`. Boundary tests for `_enforce_length` at lengths 0, 1, 4, 5, 6, 10; assert padded char == `CAPTCHA_CHARSET[0] == 'A'` when shorter.
    - **3.7 Train/val split determinism** — call `create_crnn_datasets(seed=42)` twice, assert filename lists identical for both train and val splits. Assert `len(val) / (len(train) + len(val)) ≈ 0.15 ± tol` on `data/metadata.csv` (754 → 641/113).
    - **3.8 Solver API & eval format** — with a randomly-initialised checkpoint saved via `save_crnn`, assert `CRNNCaptchaSolver.solve(path)` returns `str` of length 5; `solve_with_confidence` returns `(str, float)` with float in `[0,1]`; `solve_batch(paths)` returns list of length `len(paths)`; same for `solve_batch_with_confidence`. Smoke run `eval_crnn.evaluate` over 10 images and assert output format contains "Exact match", "CER", "Per-position accuracy", "Top-10 confusions", "Verdict".
    - **3.9 No synthetic data default** — static check: `grep -n "synthetic_crnn|generate_synthetic_crnn" train_crnn.py` matches only inside `--use-synthetic` flag handler / `if use_synthetic:` branch. Runtime: `python train_crnn.py --epochs 1` with `data/synthetic_crnn` absent/renamed completes without `FileNotFoundError`.
    - **3.10 Hardware envelope** — after 1-epoch smoke on RTX 3060, `torch.cuda.max_memory_allocated() < 8 * 1024**3` AND `psutil.Process().memory_info().rss` peak `< 16 * 1024**3` at `DEFAULT_BATCH_SIZE = 32`.
  - Observe baseline on UNFIXED code first: capture the actual values produced (charset bytes, ONNX file headers, split filename lists with `seed=42`, decode outputs on canned logits, ImageNet mean/std on synthesised images). Bake those observed oracles into the property assertions.
  - Run all preservation tests on UNFIXED code:
    - **EXPECTED OUTCOME**: All preservation tests PASS on unfixed code (this confirms the baseline behaviour we must preserve).
  - Mark task complete when tests are written, run, and all passing on unfixed code.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 3. Fix for CRNN+CTC collapse, augmentation overfit, CPU-bound DataLoader, and doc/code drift

  - [x] 3.1 Apply fix A — tone down augmentation in `dataset_crnn.py`
    - Edit `_build_albu_aug(strong=True)` and `_TV_TRAIN_AUG` per design §A:
      - Affine: `rotate=(-12,12) → (-4,4)`, `shear=(-5,5) → (-2,2)`, `translate_percent=(-0.06,0.06) → (-0.03,0.03)`, `scale=(0.85,1.15) → (0.92,1.08)`, `p=0.6 → 0.5`.
      - Perspective: `scale=(0.02,0.08) → (0.01,0.04)`, `p=0.3 → 0.2`.
      - RandomBrightnessContrast: `brightness_limit 0.2 → 0.15`, `contrast_limit 0.2 → 0.15`.
      - HueSaturationValue: `hue_shift_limit 10 → 5`, `sat_shift_limit 20 → 12`, `val_shift_limit 15 → 10`.
      - GaussNoise: `var_limit (5.0,25.0) → (3.0,12.0)`, `p=0.4 → 0.3`.
      - OneOf(GaussianBlur, MotionBlur): `p=0.25 → 0.15`.
      - CoarseDropout: `max_holes 4 → 2`, `max_height/max_width 8 → 5`, `p=0.2 → 0.1`.
    - Mirror the same intensity in the torchvision fallback `_TV_TRAIN_AUG` (rotate ±4, translate 0.03, scale (0.92, 1.08), shear 2, brightness/contrast/saturation 0.15, hue 0.04).
    - Keep `_TRAIN_AUG` at module-level so it is pickleable for spawn-based DataLoader workers; if pickling issues surface during smoke, wrap in a top-level helper `def _apply_train_aug(rgb_array)`.
    - Do NOT touch `_resize_and_normalize`, `_MEAN`, `_STD`, `CRNNCaptchaDataset.__init__`, `collate_fn`, or `create_crnn_datasets` split logic.
    - _Bug_Condition: isBugCondition(run) where `augmentation == _build_albu_aug(strong=True)` (pre-fix strengths) — see design "Bug Condition" pseudocode_
    - _Expected_Behavior: Property 1 thresholds on `eval_exact_match`, `eval_loss`, gap — see design "Correctness Properties" Property 1_
    - _Preservation: 3.1 charset, 3.2 input shape, 3.7 split logic remain unchanged — see design "Preservation Requirements"_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.7_

  - [x] 3.2 Apply fix B — hyperparam + DataLoader workers in `train_crnn.py`
    - Lower `DEFAULT_LR: float = 1e-3 → 5e-4` (design §B item 1).
    - Replace `warmup_steps = min(WARMUP_STEPS, total_steps // 10)` with `warmup_steps = max(WARMUP_STEPS, steps_per_epoch * 2)` to guarantee ≥ 2 epochs of warmup; keep `WARMUP_STEPS = 200` as floor (design §B item 2).
    - Add CLI flag `--num-workers` (default `None` → auto). Auto policy:
      - On Windows (`os.name == "nt"`): `num_workers = min(4, os.cpu_count() // 2)`; if albumentations import fails, fall back to `0` (torchvision path).
      - On Linux/macOS: `num_workers = min(8, os.cpu_count() // 2)`.
    - When `num_workers > 0`: set `persistent_workers=True` AND `prefetch_factor=4`. Keep `pin_memory=use_amp`.
    - Confirm DataLoader construction sits inside `if __name__ == "__main__":` for Windows spawn-safety; verify `_TRAIN_AUG` pickles under spawn.
    - Replace the stale comment `num_workers = 0  # Windows: 0 để tránh pickle issue với albumentations` with a comment that describes the new auto policy and notes that albumentations 1.4.3 + module-level Compose pickle correctly under Windows spawn.
    - Add `logger.info(f"DataLoader: num_workers={num_workers}, persistent_workers=…, prefetch_factor=…")` so future regressions are visible.
    - Do NOT change `DEFAULT_EPOCHS=200`, `DEFAULT_BATCH_SIZE=32`, `GRAD_CLIP_NORM=5.0`, checkpoint paths, ONNX export step, or resume payload keys.
    - _Bug_Condition: isBugCondition(run) where `base_lr == 1e-3` AND `num_workers == 0` AND `warmup_steps == 200`; CTC collapse + GPU < 80% util branches_
    - _Expected_Behavior: Property 1 — `min(eval_loss) < 3.219`, gap ≤ 0.50, mean GPU util ≥ 0.80 in 30s steady-state window_
    - _Preservation: 3.4 resume flow, 3.5 Windows boot, 3.10 hardware envelope unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.4, 3.5, 3.10_

  - [x] 3.3 Apply fix C — architecture canonicalization in `crnn_model.py`
    - Keep `CRNN.__init__(hidden_size: int = 128)` as-is (this is canonical — `count_parameters() == 2_186_553` per `train_log.txt`).
    - Fix the in-file docstring at line ~25 that says `BiLSTM: 2 layers, hidden=256, dropout 0.2 → (T=80, B, 512) (bidir = 2*hidden)` to `hidden=128 → (T=80, B, 256)`. Update the `(B, 512, h≈4, w=80)` and `(T=80, B, 512)` mentions to match `hidden=128`.
    - Add a banner comment immediately above `class CRNN`: `# Canonical params: count_parameters() == 2_186_553 with hidden_size=128. Update PIPELINE_SUMMARY.md / README.md / CLAUDE.md if you change this.`
    - Do NOT touch `CAPTCHA_CHARSET`, `NUM_CLASSES`, `CTC_BLANK_INDEX`, `CHAR_TO_IDX`, `IDX_TO_CHAR`, `INPUT_HEIGHT`, `INPUT_WIDTH`, forward/encode/decode/save/load/`export_onnx` logic.
    - _Bug_Condition: isBugCondition(run) `doc_drift` branch — `hidden_size` quoted in docs ≠ value in source code_
    - _Expected_Behavior: Property 3 — single source of truth in source code; all docs reference or quote canonical values_
    - _Preservation: 3.1 charset, 3.2 input shape, 3.3 ONNX contract, 3.6 decode semantics unchanged_
    - _Requirements: 2.5, 3.1, 3.2, 3.3, 3.6_

  - [x] 3.4 Apply fix D — doc/code reconciliation across 6 docs
    - Edit `README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`: in hyperparam tables replace `BiLSTM hidden=256` with `BiLSTM hidden=128 (see crnn_model.CRNN default)`; replace `Epochs=50` with `Epochs=200 (see train_crnn.DEFAULT_EPOCHS)`; quote `count_parameters() == 2,186,553`. Note the new `DEFAULT_LR=5e-4` and the auto `num_workers` policy.
    - Edit `docs/adr/0001-crnn-ctc-over-softmax.md`: append an "Implementation note (2026-05-15)" paragraph stating canonical `hidden_size=128` (not 256 as in original draft) and verified param count `2,186,553` from `CRNN().count_parameters()`.
    - Edit `docs/codebase/ARCHITECTURE.md` "Known Architectural Risks": change the `Doc/code drift on hyperparameters` bullet to `Resolved (2026-05-15): canonical hidden_size=128, DEFAULT_EPOCHS=200; docs synced.`
    - Edit `docs/codebase/CONCERNS.md` "Top Risks" entry "Documentation/code drift on the BiLSTM hidden size and default epochs": set status `Resolved` with reference to the fix commit.
    - Do NOT modify `CONTEXT.md`, `docs/research_strategy_*`, or any doc outside this 6-file set.
    - _Bug_Condition: isBugCondition(run) `doc_drift` branch_
    - _Expected_Behavior: Property 3 — single source of truth; doc-drift grep returns empty after fix_
    - _Preservation: All non-listed docs untouched; all non-hyperparam doc content unchanged_
    - _Requirements: 2.5_

  - [x] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Trained CRNN+CTC Generalises To Real Val And GPU Saturated
    - **IMPORTANT**: Re-run the SAME tests from task 1 — do NOT write new tests.
    - The tests from task 1 encode the expected behavior. When they pass, they confirm the expected behaviour is satisfied.
    - Sequence:
      1. Run `pytest tests/test_property1_bug_condition.py::test_property1_smoke` — **EXPECTED OUTCOME**: PASSES (`max(eval_exact_match) ≥ 0.05`, `min(eval_loss) < 3.0`, gap ≤ 0.7) on the post-fix code.
      2. Run `pytest tests/test_property1_bug_condition.py::test_property1_gpu_util` — **EXPECTED OUTCOME**: PASSES (mean GPU util ≥ 0.80 in 30s steady-state window).
      3. Run `pytest -m slow tests/test_property1_bug_condition.py::test_property1_full` — **EXPECTED OUTCOME**: PASSES (`max(eval_exact_match) ≥ 0.30` AND `min(eval_loss) < log(25)` AND `min(eval_loss) < 3.388` AND gap ≤ 0.50).
      4. Run doc-drift helper — **EXPECTED OUTCOME**: now FAILS to find any `hidden=256` / `Epochs=50` matches across the 6 doc files (i.e. drift removed).
    - If any sub-test still fails, do NOT silently relax thresholds; instead re-examine which sub-fix (A/B/C/D) is incomplete and update only the implementation, not the test.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - All Non-Training-Tuning Behavior Identical
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests.
    - Run `pytest tests/test_property2_preservation.py -v` covering all 10 preservation invariants (3.1 – 3.10).
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions across charset, input shape, ONNX contract, resume flow, Windows boot, decode semantics, split determinism, Solver API, no-synthetic default, hardware envelope).
    - Confirm all tests still pass after fix; if any preservation test fails, treat it as a regression and revert/repair the offending sub-fix before continuing.

- [x] 4. Checkpoint - Ensure all tests pass
  - Run the full test suite: `pytest tests/test_property1_bug_condition.py tests/test_property2_preservation.py -v` (slow tests opted-in via `-m slow` for the full 200-epoch validation when on the hardware target).
  - Confirm:
    - All Property 1 sub-tests PASS on fixed code (smoke + full + GPU util).
    - All Property 2 preservation invariants PASS on fixed code.
    - Doc-drift grep returns empty across the 6 reconciled docs.
  - Spot-check artefacts produced by the run:
    - `captcha_crnn_model.onnx` exists with correct I/O contract.
    - `captcha_crnn_last.pth` resumes cleanly via `python train_crnn.py --resume --epochs 1`.
    - `python eval_crnn.py` produces the expected verdict format.
  - Ask the user if any unexpected regressions or threshold ambiguities arise (e.g. Property 1 smoke barely meets `0.05` floor — clarify whether to escalate to full 200-epoch run or to re-tune one of the four sub-fixes).


## Notes

- **Property numbering**: Per the bugfix workflow contract, Property 1 is reserved for the Bug Condition exploration test (task 1) and is re-used as `Property 1: Expected Behavior` in sub-task 3.5 (same test, re-run on fixed code). Property 2 is the Preservation property covered by tasks 2 and 3.6. Property 3 from the design (Doc/Code Reconciliation) is folded into Property 1's `doc_drift` branch and is verified by the doc-drift helper inside task 1 / sub-task 3.5.
- **Scoped PBT for deterministic bug**: The bug is deterministic given the buggy config (`data/metadata.csv`, `seed=42`, hardware target, pre-fix hyperparams). The exploration test scopes the property to that single concrete failing case; the "for-all" generalisation is over re-runs of that config.
- **Test cost**: The full Property 1 verification requires a 200-epoch training run on the hardware target (≈ 6 minutes on RTX 3060 per `train_log.txt`). The smoke variant (20 epochs, ≈ 1 minute) is the primary quick gate; the full variant is gated behind `pytest -m slow` and run as the final correctness verification.
- **Hardware-dependent tests**: Property 1 GPU-util sub-test and the Windows multi-worker boot smoke (3.5) require Windows + RTX 3060. They `pytest.skip` with a clear reason on other hosts so the suite still runs on CI / dev boxes.
- **Out of scope (per bugfix.md)**: No synthetic data generation; no hardware target changes; no charset / input shape / ONNX contract / Solver API changes. Any deviation surfaces as a Property 2 preservation failure and must be reverted.
- **Doc-drift assertion polarity**: On unfixed code the doc-drift helper PASSES (matches found → bug confirmed). After fix D, the same helper FAILS (no matches → drift removed). Sub-task 3.5 explicitly inverts the assertion polarity for the post-fix run.
