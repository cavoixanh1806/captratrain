# Testing Patterns

## Core Sections (Required)

There is no test framework, no automated test suite, and no coverage tooling configured in this repository. The closest thing to tests are (a) per-module `if __name__ == "__main__":` smoke blocks, (b) the `run_smoke.bat` end-to-end smoke pipeline (5 epochs, batch 16, no augmentation), and (c) ad-hoc verification via `eval_crnn.py` against the labelled real corpus. This document records that explicitly so the gap is not mistaken for a missing-but-existing setup.

### 1) Test Stack and Commands

- Primary test framework: none. Verified — no `pytest.ini`, `pyproject.toml [tool.pytest]`, `tox.ini`, `unittest` discovery, `tests/`, `__tests__/`, `spec/`, or `test_*.py` files. (`grep_search` over `*.py` for `^if __name__` enumerates only run-as-script entry points; no test imports.)
- Assertion/mocking tools: none.
- Commands (closest equivalents):

```bash
# Module-level smoke checks (each prints shapes/sample output and exits)
python crnn_model.py        # CRNN forward pass + encode/decode sanity
python dataset_crnn.py      # CRNNCaptchaDataset[0] + collate_fn shape check
python synthetic_renderer.py  # render 5 captchas to <tempdir>/captra_smoke

# Pipeline smoke (CPU friendly, ~30-60 minutes)
run_smoke.bat               # train 5 epochs batch 16 no-augment + eval

# Ground-truth evaluation (closest to a regression test)
python eval_crnn.py
```

### 2) Test Layout

- Test file placement pattern: not applicable. There are no test files.
- Naming convention: not applicable.
- Setup files and where they run: not applicable.

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | no | none | The `__main__` smoke blocks exercise shapes and a single `__getitem__` call but do not assert against expectations. They print and exit. |
| Integration | partial (manual) | `dataset_crnn → train_crnn → eval_crnn` end-to-end | `run_smoke.bat` runs the full training and evaluation pipeline at minimum settings to catch crashes; success is judged by a human reading `smoke_log.txt`. |
| E2E | partial (manual) | Full pipeline plus inference | `run_all.bat` runs `import_new_data.py → train_crnn.py → eval_crnn.py`. `eval_crnn.evaluate` prints a verdict (`EXCELLENT`/`GOOD`/`OK`/`FAIL`) against the 90% target. There is no automated assertion or CI gate on that verdict. |
| Property-based | no | none | None configured. |

### 4) Mocking and Isolation Strategy

- Main mocking approach: none. No `unittest.mock`, `pytest-mock`, or `responses` usage anywhere.
- Isolation guarantees: training rewrites `captcha_crnn_model.pth`/`captcha_crnn_last.pth`/`captcha_crnn_model.onnx` at the project root; `run_all.bat` and `run_smoke.bat` delete these before training to avoid stale state. Synthetic generation calls `shutil.rmtree(output_dir)` before regenerating (`generate_synthetic_crnn.generate`).
- Common failure mode: hidden coupling on the working directory. Several modules use relative paths (`data/`, `data/metadata.csv`, `data/synthetic_crnn/`, `dataset/`, `captcha_crnn_*.pth`). Running from any directory other than the repo root will fail.

### 5) Coverage and Quality Signals

- Coverage tool + threshold: `[TODO]` not configured. No `coverage.py`, no `.coveragerc`, no `[tool.coverage]` table.
- Current reported coverage: `[TODO]` not measured.
- Known gaps/flaky areas:
  - No unit tests for `crnn_model.encode_label` invalid-character path, `decode_greedy_with_confidence` empty-output handling, or `inference_crnn._enforce_length` boundary behavior.
  - No tests for `import_new_data.py` dedupe logic (duplicate label handling silently drops new images — see `CONCERNS.md`).
  - The `run_smoke.bat` "did the loss go down?" check is human-only.
  - `eval_crnn.evaluate` produces a verdict string but the `.bat` orchestrators do not parse it; a regression that drops accuracy below 90% will not fail the pipeline.

### 6) Evidence

- Repo-wide search for test artifacts: no matches for `pytest.ini`, `tests/`, `test_*.py`, `conftest.py`, `unittest`.
- `docs/codebase/.codebase-scan.txt` (PERFORMANCE & TESTING: "No performance testing configs detected"; CI/CD: "No CI/CD pipelines detected").
- `crnn_model.py`, `dataset_crnn.py`, `synthetic_renderer.py` (the `__main__` smoke blocks).
- `run_smoke.bat`, `run_all.bat` (manual pipeline checks).
- `eval_crnn.py` (verdict logic).
