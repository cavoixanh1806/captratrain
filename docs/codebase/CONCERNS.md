# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| resolved | Documentation/code drift on the BiLSTM hidden size and default epochs | `crnn_model.py` `class CRNN(... hidden_size: int = 128)`; `train_crnn.py` `DEFAULT_EPOCHS: int = 200`; previously `README.md`/`PIPELINE_SUMMARY.md`/`CLAUDE.md` described BiLSTM `hidden` and default epoch count at stale values; intent docs quoted `~2.18M params` | Resolved (2026-05-15) by the `crnn-ctc-collapse-fix` bugfix spec (fix D, doc/code reconciliation across the 6 doc set). Source code is now the single source of truth; all six docs reference the canonical constants by name and quote `count_parameters() == 2,186,553`. | Status: **Resolved**. See `.kiro/specs/crnn-ctc-collapse-fix/` (sub-task 3.4 — "Apply fix D — doc/code reconciliation across 6 docs") and the corresponding fix commit. |
| high | No automated test suite or CI gate | scan: `CI/CD PIPELINES → No CI/CD pipelines detected`; no `tests/` or `pytest.ini` | A regression in `decode_greedy`, `encode_label`, `_enforce_length`, or `create_crnn_datasets` would not be caught until manual `run_all.bat` execution | Add minimal `pytest` for charset round-tripping, CTC decode, length enforcement, dedupe in `import_new_data`, and metadata split logic. Optionally fail the orchestrators when `eval_exact_match` falls below a threshold |
| medium | `dataset/` import dedupes by label only | `import_new_data.py` `existing_labels = set(df["text"]....)`; comment "skip (anh da import truoc do)" | Two distinct real captchas with the same label silently lose the newer one — gradually erodes the real corpus | Dedupe by perceptual hash or by source filename; or accept duplicates and only de-dup on filename collision |
| medium | `_enforce_length` pads with `'A'` for short outputs | `inference_crnn._enforce_length` (`return text + CAPTCHA_CHARSET[0] * (target - len(text))`) | Guarantees a wrong character whenever the CTC decode is short; CER reported by `eval_crnn` is biased upward in those cases | Replace with a length-aware decoder (e.g. CTC beam search + min/max length constraint) or surface the short-prediction case as a low-confidence flag |
| medium | Race on `data/_crnn_train.csv` and `data/_crnn_val.csv` | `dataset_crnn.create_crnn_datasets` writes both files with fixed names, then `unlink(missing_ok=True)` in the same call | Two parallel training runs in the same working directory clobber each other's split files | Switch to `tempfile.NamedTemporaryFile` or build the split entirely in-memory by passing a DataFrame into `CRNNCaptchaDataset` |
| medium | `label_server.py` binds `0.0.0.0:8080` with no auth and a hardcoded absolute `DATA_DIR` | `label_server.py` (`HOST = "0.0.0.0"`, `PORT = 8080`, `DATA_DIR = Path(r"C:\Users\Administrator\Desktop\captratrain\data")`) | Anyone on the LAN can read all captcha images and overwrite `metadata.csv`. Hardcoded path makes the tool non-portable to other machines | Bind `127.0.0.1` by default, gate `POST /api/save` behind a token, and resolve `DATA_DIR` from a CLI arg or relative path |
| medium | Eight-thousand+ tracked items in repo, dominated by data | scan: `Total files scanned: 8029`; `Files by language: Other 8017, Python 12`; `data/synthetic_crnn/captcha_*.png` listed in top-10 largest | Slow git operations, slow IDE indexing; risk of accidentally committing synthetic images | `.gitignore` already excludes `data/synthetic_crnn/` and `dataset/`; verify no synthetic PNGs are tracked (`git ls-files | findstr synthetic_crnn`); add a `.gitattributes`/git-lfs policy if real captchas grow further |
| low | Long single-file responsibilities | `train_crnn.py` 17.7 KB, `synthetic_renderer.py` 22.8 KB, `system_info.py` 23.5 KB, `self_train.py` 14.9 KB, `label_server.py` 17.8 KB | Harder to navigate and review. Acceptable for a research-scale repo but would benefit from light decomposition | Split when adding the next feature (e.g. extract `train_crnn.compute_metrics`/`_edit_distance` into a `metrics.py` shared with `eval_crnn`) |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| `eval_crnn` imports a private helper | `eval_crnn.py` `from train_crnn import _edit_distance` | `eval_crnn.py:15`, `train_crnn.py:_edit_distance` | Coupling between training and evaluation modules; renames break consumers silently | Move `_edit_distance` and `compute_metrics` into a shared `metrics.py` module |
| `self_train` re-imports training helpers including private ones | `self_train.py` imports `train_one_epoch`, `validate`, `build_warmup_cosine_scheduler`, `compute_metrics`, plus the path constants | `self_train.py:25-32` | Hard to refactor `train_crnn.py` without breaking `self_train` | Same as above — share via `training_steps.py` and `paths.py` modules |
| Hyperparam values declared in three places | `crnn_model.CRNN(hidden_size=128)`, `train_crnn.DEFAULT_EPOCHS=200`/`DEFAULT_BATCH_SIZE=32`, README/PIPELINE_SUMMARY/CLAUDE markdown tables | Multiple files | Drift (already happened on `hidden_size` and `epochs`) | Single source of truth in code; doc tables should reference the constants by name |
| `os` imported but unused | `synthetic_renderer.py` `import os`, `label_server.py` `import os` | both files | Cosmetic; suggests removed code paths | Remove unused imports (a one-line cleanup once a linter is added) |
| `Sequence` imported but unused | `crnn_model.py` `from typing import Sequence` | `crnn_model.py:21` | Cosmetic | Remove unused import |
| Vietnamese inline comments mixed with English docstrings | Most modules | repo-wide | Onboarding friction for non-Vietnamese readers; not a bug | `[ASK USER]` decide a project language convention; if mixed is fine, document it in `CONVENTIONS.md` |
| Module-level `random` calls without seeding | `synthetic_renderer.random_text`, `random_text_color_hsv`, etc. seed only the global `random` | `synthetic_renderer.py` | Synthetic generation is non-reproducible run-to-run | Accept a `--seed` arg in `generate_synthetic_crnn.py` and call `random.seed(seed)` + `np.random.seed(seed)` at the top |
| ONNX deployment path described in README but never imported | `README.md` `import onnxruntime as ort` snippet; no source module imports `onnxruntime` | docs only | Risk that the deploy snippet drifts from the trained model's input contract | Add a tiny `verify_onnx.py` that loads the exported `.onnx` and asserts the (1, 3, 64, 320) input contract |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Unauthenticated label server on LAN | A01 Broken Access Control | `label_server.py` (`HOST="0.0.0.0"`, no auth on `POST /api/save`) | None | Bind to `127.0.0.1` by default; require a shared token for write endpoints |
| Path traversal via `GET /images/{name}` | A01 Broken Access Control | `label_server._send_image` constructs `DATA_DIR / filename` from the URL path with only `path.replace("/images/", "")` | The `Path` join keeps the result inside `DATA_DIR` for normal inputs but does not actively reject `..\..\` segments | Validate that the resolved path is under `DATA_DIR` (`filepath.resolve().is_relative_to(DATA_DIR.resolve())`) before opening |
| Untrusted `torch.load` of checkpoints | n/a (supply chain) | `crnn_model.load_crnn` and `train_crnn.main` call `torch.load(path, ...)` without `weights_only=True` | Checkpoints are produced locally; not currently downloaded from the network | Pass `weights_only=True` once on PyTorch ≥ 2.4 to harden against malicious checkpoint files; document that consumers should not load untrusted `.pth` files |
| ONNX opset 14 without dependency floor | n/a | `crnn_model.export_onnx(..., opset_version=14)`; `requirements.txt` `onnx>=1.15`, `onnxruntime>=1.17` | Lower bound only | Pin upper bound in production deploys; record opset requirement in deployment docs |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| `num_workers=0` in DataLoader | `train_crnn.py` (`num_workers = 0  # Windows: 0 để tránh pickle issue với albumentations`); same in `self_train.py` | Single-threaded data loading; CPU-bound preprocessing during GPU steps | Likely under-utilises a fast GPU during training; widens the gap between RTX 3060 and RTX 5090 estimates in the README | On Linux, set `num_workers > 0`; on Windows, refactor augmentation to a top-level `Compose` and re-test workers |
| Synthetic generation is single-threaded | `generate_synthetic_crnn.generate` is a plain `for i in range(count)` loop | At 100K samples this is hours of CPU work | Bigger corpora become impractical | Use `concurrent.futures.ProcessPoolExecutor` over `render_text_on_image`; fonts and palettes are read-only |
| Validation reads the full image set every epoch | `train_crnn.validate` iterates the val DataLoader (no caching of decoded tensors) | At 754 × 0.15 ≈ 114 images this is fine; at larger corpora it grows linearly | Future scaling pain when val grows | Cache preprocessed val tensors in memory after the first pass |
| `inference_crnn.solve_batch` re-reads from disk per call | `_read_image` invoked inside the batch loop | Acceptable for one-shot inference, suboptimal for evaluation runs that call it twice | Larger eval corpora double their disk I/O | Add a path-keyed in-memory cache or expose a tensor-input API |
| `label_server.write_metadata` rewrites the whole CSV on every save | `label_server._save_label` reads, mutates, writes the entire file each `POST` | Fine at 754 rows; O(n) per click | Slows down with very large datasets | Append-only journal plus periodic compaction, or hold the CSV in memory and flush on shutdown |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| Removed predecessors (`train.py`, `inference.py`, `dataset.py`, `train_unet.py`, `generate_unet_data.py`) appear in the high-churn list but no longer exist | The repo went through a TrOCR → CRNN rewrite (commit `bf743f3`). Recent history is dominated by deleted files | scan `HIGH-CHURN FILES` top entries are gone (`27 README.md`, `15 train.py`, `9 inference.py`, `8 generate_unet_data.py`, `6 dataset.py`) | Treat the 90-day churn list as historical context, not a live fragility map. Real current hot spots are `train_crnn.py`, `crnn_model.py`, `dataset_crnn.py`. |
| `crnn_model.py` and `train_crnn.py` recently modified together | Latest commit "Optimize CRNN model to 2.18M params, increase epochs to 200, and improve data augmentation" | both files updated `2026-05-15 04:02 PM` | Any architecture change must update three call sites: `crnn_model.CRNN`, `train_crnn.main` defaults, and `inference_crnn` (which loads via `crnn_model.load_crnn`). |
| `synthetic_renderer.py` — calibration tuned for 754 specific images | A drift between the synthetic distribution and the real distribution silently degrades round-1 training | recent commits "BG synthetic calibrated tu 754 anh real", "fix: bo real BG extraction" | When the real corpus grows, re-run calibration analysis before regenerating synthetic data; pin the calibration parameters in a comment block at the top of the file. |
| `run_all.bat` "tee via inline `python -c`" pattern | Python 3.10+ comprehension with side effects (`[(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]`) is a deliberate workaround for `Tee-Object` UTF-8 bugs | recent commit "Fix Windows Tee-Object encoding bug" | Do not "modernize" the inline tee without testing on Windows CMD/PowerShell with non-ASCII characters in logs. |

### 6) `[ASK USER]` Questions

1. [ASK USER] Which BiLSTM hidden size is canonical — `128` (current code in `crnn_model.CRNN.__init__`) or `256` (README, PIPELINE_SUMMARY, CLAUDE.md, ADR-0001)? The `~2.18M params` claim in docs cannot be true for both.
2. [ASK USER] Which default epoch count is canonical — `200` (`train_crnn.DEFAULT_EPOCHS`) or `50` (README, PIPELINE_SUMMARY hyperparam tables)?
3. [ASK USER] What is the project language policy for new comments and docstrings — Vietnamese, English, or mixed (current state)? Need this to lock down `CONVENTIONS.md`.
4. [ASK USER] Should `import_new_data.py` keep deduping by label, switch to perceptual-hash dedup, or accept duplicates? The current behavior is fine for the original 500-image bootstrap but loses real captchas as the corpus grows.
5. [ASK USER] Is `label_server.py` ever exposed beyond a single trusted machine? If yes, the unauthenticated `0.0.0.0:8080` binding and the path-traversal-prone `GET /images/...` handler need fixing.
6. [ASK USER] Should `eval_crnn.evaluate`'s 90% verdict become a hard gate in `run_all.bat` (non-zero exit on FAIL), or stay informational?
7. [ASK USER] Is the ONNX runtime deploy path in `README.md` actually used in production, or is it aspirational? If used, where, and should the project ship a `verify_onnx.py` smoke check?

### 7) Evidence

- Scan output sections: `TODO / FIXME / HACK` (only false positives in `scan.py` source), `HIGH-CHURN FILES`, `CODE METRICS`, `CI/CD PIPELINES`, `SECURITY & COMPLIANCE`, `LINTING AND FORMATTING CONFIG`, `ENVIRONMENT VARIABLE TEMPLATES`
- `crnn_model.py`, `train_crnn.py`, `dataset_crnn.py`, `inference_crnn.py`, `self_train.py`, `eval_crnn.py`, `import_new_data.py`, `label_server.py`, `synthetic_renderer.py`, `generate_synthetic_crnn.py`, `system_info.py`
- `README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`, `CONTEXT.md`
- `docs/adr/0001-crnn-ctc-over-softmax.md`, `docs/adr/0002-synthetic-first-training.md`
- `run_all.bat`, `run_smoke.bat`, `setup.bat`
- `.gitignore`
