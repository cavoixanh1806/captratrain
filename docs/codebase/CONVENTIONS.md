# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | `snake_case.py` for source modules; `_crnn` suffix for CRNN-pipeline modules; ADRs `<NNNN>-<kebab-title>.md` | `dataset_crnn.py`, `train_crnn.py`, `docs/adr/0001-crnn-ctc-over-softmax.md` | repo root listing; `docs/adr/` |
| Classes | `PascalCase` | `class CRNN(nn.Module)`, `class ConvBlock(nn.Module)`, `class CRNNCaptchaDataset(Dataset)`, `class CRNNCaptchaSolver`, `class LabelHandler(SimpleHTTPRequestHandler)` | `crnn_model.py`, `dataset_crnn.py`, `inference_crnn.py`, `label_server.py` |
| Functions / methods | `snake_case`; module-private helpers prefixed `_` | `encode_label`, `decode_greedy_with_confidence`, `_edit_distance`, `_preprocess`, `_resize_and_normalize` | `crnn_model.py`, `inference_crnn.py`, `train_crnn.py` |
| Constants | `UPPER_SNAKE_CASE` at module scope | `CAPTCHA_CHARSET`, `NUM_CLASSES`, `CTC_BLANK_INDEX`, `INPUT_HEIGHT`, `INPUT_WIDTH`, `DEFAULT_EPOCHS`, `CHECKPOINT_PATH`, `LAST_CHECKPOINT_PATH` | `crnn_model.py`, `train_crnn.py` |
| Module-private constants | `_UPPER_SNAKE_CASE` or `_lower_snake_case` for tunables not part of the public API | `_MEAN`, `_STD`, `_TRAIN_AUG`, `_TV_TRAIN_AUG`, `_HAS_ALBU` | `dataset_crnn.py` |
| Dataset filename pattern | Real images `map_<NNNNN>.png` (5-digit); synthetic images `captcha_<NNNNNN>.png` (6-digit); inbox images `map_<LABEL>.png` (5 chars in charset) | `data/map_00000.png`, `import_new_data.py` regex `^map_(\d+)\.png$` and `^map_([A-Z0-9]+)\.png$` | `import_new_data.py`, `generate_synthetic_crnn.py`, scan tree |

### 2) Formatting and Linting

- Formatter: None configured. No `pyproject.toml`, no `.editorconfig`, no `black`/`ruff` configs. (scan output: "No linting or formatting config files found in project root")
- Linter: None configured.
- Observed style is consistent across modules even without enforcement:
  - 4-space indentation, line length informally ~90-100 chars.
  - Triple-double-quoted module and function docstrings (Google-ish: `Args`, `Returns`, sometimes `Raises`).
  - Vietnamese inline comments are common; English is used for public docstrings (mixed in places).
  - Type hints on most function signatures, including PEP 604 union syntax (`int | tuple[int, int]`, `str | Path`, `dict | None`).
  - `from __future__ import annotations` only in `system_info.py`; other modules rely on the Python 3.10+ baseline.
- Run commands: not applicable. `[TODO]` add a formatter and linter (suggested: `ruff format` + `ruff check`) and decide whether to enforce in CI.

### 3) Import and Module Conventions

- Imports are direct module references; there are no path aliases or barrels. Example: `from crnn_model import CRNN, decode_greedy, save_crnn` (`train_crnn.py`).
- Standard ordering observed: stdlib first, then third-party (`torch`, `cv2`, `numpy`, `pandas`, `PIL`), then local modules.
- Local imports use module names (no relative `.foo` imports because everything is at the repo root).
- Optional dependencies are guarded by `try / except ImportError` and a sentinel boolean (`_HAS_ALBU` in `dataset_crnn.py`; `try: import psutil` in `system_info.py`; `try: import torch` in `system_info.py`).
- Local imports inside functions appear when delaying heavy modules until needed: `from torch.utils.data import ConcatDataset` inside `dataset_crnn.create_crnn_datasets`; `from tqdm import tqdm` inside `train_crnn.train_one_epoch` and `validate`.

### 4) Error and Logging Conventions

- Logging: every entry-point module configures the root logger with `logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")` and uses a module-level `logger = logging.getLogger(__name__)`. Files: `eval_crnn.py`, `train_crnn.py`, `self_train.py`, `inference_crnn.py`, `import_new_data.py`, `generate_synthetic_crnn.py`, `synthetic_renderer.py`.
- Direct `print()` is also used for human-facing reports (`eval_crnn.evaluate` ASCII tables and verdict; `train_crnn.train_one_epoch` Hugging Face style log dicts via `tqdm.write`).
- Error strategy by layer:
  - Data layer raises explicit exceptions with actionable hints — `FileNotFoundError(f"Real metadata not found: {meta_real}. Run label_server.py + import_new_data.py first.")`, `ValueError("Phải bật ít nhất một trong use_real / use_synthetic")`.
  - `crnn_model.encode_label` raises `ValueError` for out-of-charset characters.
  - `inference_crnn.CRNNCaptchaSolver.__init__` raises `FileNotFoundError` with a "Hãy train trước: python train_crnn.py" hint.
  - Tooling/best-effort failures (ONNX export at the end of training, ONNX re-export in `self_train`) are caught and logged as warnings — `logger.warning(f"ONNX export failed (non-fatal): {e}")`.
  - Subprocess helpers in `system_info._run` swallow `TimeoutExpired`, `FileNotFoundError`, and `OSError` and return `None`.
- Sensitive data: there are no credentials or PII handled by this codebase. Logs do not contain secrets.

### 5) Testing Conventions

- Test file naming/location: not applicable — no `tests/`, no `test_*.py`, no `pytest.ini` or `conftest.py` in the repo. Several modules include an in-`__main__` smoke test that exercises basic shapes and IO (`crnn_model.py`, `dataset_crnn.py`, `synthetic_renderer.py`). See `TESTING.md` for details.
- Mocking strategy norm: `[TODO]` no test framework or mocks present.
- Coverage expectation: `[TODO]` not configured.

### 6) Evidence

- `crnn_model.py`, `dataset_crnn.py`, `train_crnn.py`, `inference_crnn.py`, `self_train.py`, `eval_crnn.py`, `import_new_data.py`, `system_info.py`
- `docs/codebase/.codebase-scan.txt` (LINTING AND FORMATTING CONFIG section)
- `docs/agents/domain.md` (project glossary expectation)
