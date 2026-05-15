# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

The repository is a flat single-package Python project. All source modules sit at the repo root; there is no `src/` directory and no monorepo workspace.

| Path | Purpose | Evidence |
|------|---------|----------|
| `crnn_model.py` | CRNN architecture, charset constants, CTC encode/decode helpers, save/load, ONNX export | file contents |
| `dataset_crnn.py` | `CRNNCaptchaDataset`, augmentation pipelines (albumentations + torchvision fallback), CTC `collate_fn`, `create_crnn_datasets` train/val factory | file contents |
| `train_crnn.py` | Training loop (200 epochs default), AMP, warmup-cosine scheduler, best/last checkpointing, ONNX export at end | file contents |
| `eval_crnn.py` | Evaluation: exact match, CER, per-position accuracy, top-10 confusions, confidence stats, verdict | file contents |
| `inference_crnn.py` | `CRNNCaptchaSolver` class (single + batch + with-confidence) and CLI entry point | file contents |
| `self_train.py` | Round-2 pseudo-labeling: predict all real images, keep predictions where conf ≥ threshold AND match ground truth, fine-tune at low LR | file contents |
| `generate_synthetic_crnn.py` | CLI to render `--count` synthetic captchas to `data/synthetic_crnn/` plus metadata.csv | file contents |
| `synthetic_renderer.py` | Calibrated background and per-character rendering matched to 754 real samples | file contents |
| `import_new_data.py` | Copy `dataset/map_<LABEL>.png` files into `data/` with sequential names, append to `metadata.csv`, dedupe by label | file contents |
| `label_server.py` | Local HTTP server on `0.0.0.0:8080` serving an HTML labeling UI with REST endpoints | file contents |
| `system_info.py` | OS/CPU/RAM/Disk/GPU/Python report (Markdown or JSON) plus training-readiness verdict | file contents |
| `requirements.txt` | Production deps (torch installed separately) | file contents |
| `setup.bat` / `run_smoke.bat` / `run_all.bat` | Windows orchestrators for setup, smoke test, and full pipeline | file contents |
| `data/` | 754 real captcha PNGs (`map_NNNNN.png`) plus `metadata.csv`; synthetic output goes to `data/synthetic_crnn/` (gitignored) | scan tree, `.gitignore` |
| `dataset/` | Inbox for new labelled images named `map_<LABEL>.png` (gitignored) | `import_new_data.py`; `.gitignore` line `dataset/` |
| `docs/adr/` | Architecture decision records (CRNN+CTC choice, synthetic-first strategy) | `docs/adr/0001-...`, `0002-...` |
| `docs/agents/` | Agent operating instructions (issue tracker, triage labels, domain glossary pointer) | `docs/agents/*.md` |
| `docs/research_strategy_20260515.md` | Background research strategy doc | file present |
| `docs/codebase/` | This documentation set | this directory |
| `.kiro/skills/acquire-codebase-knowledge/` | Skill that generates these docs (script + templates + references) | directory listing |
| `captcha_crnn_last.pth` | Latest training checkpoint (~25 MB), gitignored | scan `Top 10 largest files` |
| `train_log.txt` / `smoke_log.txt` | Run logs written by the `.bat` orchestrators (gitignored) | `run_all.bat`, `run_smoke.bat`, `.gitignore` |
| `CLAUDE.md` / `CONTEXT.md` / `PIPELINE_SUMMARY.md` / `README.md` / `research_minecraft_map_captcha_20260515.md` | Project-level intent docs | files present |

### 2) Entry Points

- Main runtime entry (training): `train_crnn.py` — `if __name__ == "__main__"` parses CLI args and calls `main()`.
- Inference entry: `inference_crnn.py` — CLI for single-image prediction; library entry is `CRNNCaptchaSolver`.
- Evaluation entry: `eval_crnn.py` — CLI computing metrics across `data/metadata.csv`.
- Self-training entry: `self_train.py` — CLI for round-2 pseudo-labeling.
- Synthetic data entry: `generate_synthetic_crnn.py` — CLI to render N samples.
- Data import entry: `import_new_data.py` — copies `dataset/*.png` into `data/` and updates metadata.
- Labeling UI entry: `label_server.py` — long-running HTTP server (port 8080).
- System diagnostic entry: `system_info.py` — emits Markdown or JSON report.
- Smoke test for `crnn_model.py`, `dataset_crnn.py`, and `synthetic_renderer.py`: each module's `if __name__ == "__main__"` block runs an internal sanity check.
- Workflow selection: `run_all.bat` (full pipeline), `run_smoke.bat` (CPU smoke), `setup.bat` (environment bootstrap).

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `crnn_model.py` (model) | `CRNN` class, charset constants, `encode_label`, `decode_greedy`, save/load, ONNX export | Dataset I/O, training loops, file globbing |
| `dataset_crnn.py` (data) | Dataset class, augmentation pipelines, collate, real/synthetic combination | Model architecture, training step |
| `synthetic_renderer.py` (data generation) | Pure rendering functions (`random_text`, `render_text_on_image`, `get_random_real_background`) | Disk I/O for batch generation, CLI |
| `generate_synthetic_crnn.py` (CLI) | Loops over `render_text_on_image`, writes PNG + metadata.csv | Rendering logic itself |
| `train_crnn.py` / `self_train.py` (training) | Training loops, optimizers, schedulers, checkpointing | Model definition, dataset construction |
| `inference_crnn.py` (inference) | `CRNNCaptchaSolver` wrapping `load_crnn` + `decode_greedy` | Training, evaluation, file labeling |
| `eval_crnn.py` (evaluation) | Aggregate metrics, per-position breakdown, verdict reporting | Training |
| `import_new_data.py` / `label_server.py` (labeling) | metadata.csv mutation and labelling UI | Model or training logic |
| `system_info.py` (diagnostics) | Host introspection and verdict | Project-specific training logic |

Cross-cutting imports observed: `dataset_crnn` imports `INPUT_HEIGHT`, `INPUT_WIDTH`, `encode_label` from `crnn_model`; `inference_crnn` imports decode helpers and `_MEAN`/`_STD` from `dataset_crnn`; `train_crnn` imports model and dataset helpers; `self_train` imports `CHECKPOINT_PATH`, `train_one_epoch`, `validate`, `build_warmup_cosine_scheduler`, `compute_metrics` from `train_crnn`; `eval_crnn` imports `_edit_distance` from `train_crnn` and `CRNNCaptchaSolver` from `inference_crnn`; `generate_synthetic_crnn` imports from `synthetic_renderer`.

### 4) Naming and Organization Rules

- Python files: `snake_case.py`. Twelve modules at root, all underscored (e.g. `crnn_model.py`, `dataset_crnn.py`, `generate_synthetic_crnn.py`).
- Suffix `_crnn` marks the post-TrOCR migration (`crnn_model`, `dataset_crnn`, `train_crnn`, `eval_crnn`, `generate_synthetic_crnn`, `inference_crnn`). Older TrOCR/U-Net names (`train.py`, `inference.py`, `dataset.py`, `train_unet.py`, `generate_unet_data.py`) appear in `.gitignore`/git history but were removed; see commit `bf743f3` "Replace TrOCR with CRNN+CTC pipeline".
- Real data filenames: `map_<NNNNN>.png` (5-digit zero-padded sequential).
- Inbox filenames: `map_<LABEL>.png` (5-character label, A-Z and 0-9), enforced by `import_new_data.py` regex.
- Synthetic filenames: `captcha_<NNNNNN>.png` (6-digit) with label in `metadata.csv`.
- Classes: `PascalCase` (`CRNN`, `ConvBlock`, `CRNNCaptchaDataset`, `CRNNCaptchaSolver`, `LabelHandler`).
- Functions and module-level helpers: `snake_case`, with `_` prefix for module-private helpers (`_edit_distance`, `_preprocess`, `_resize_and_normalize`, `_load_random_font`, `_run`, `_md_kv_table`).
- Constants: `UPPER_SNAKE_CASE` (`CAPTCHA_CHARSET`, `INPUT_HEIGHT`, `INPUT_WIDTH`, `CTC_BLANK_INDEX`, `DEFAULT_EPOCHS`, `CHECKPOINT_PATH`).
- No path aliases; all imports are direct module-name imports relative to the repo root.
- ADRs: `docs/adr/<NNNN>-<kebab-title>.md` (currently `0001`, `0002`).

### 5) Evidence

- `docs/codebase/.codebase-scan.txt` (DIRECTORY TREE, MONOREPO SIGNALS, CODE METRICS)
- repo root `dir *.py` listing
- Source files cited above for each module
- `.gitignore` (which directories are ephemeral)
- `docs/adr/0001-crnn-ctc-over-softmax.md`, `docs/adr/0002-synthetic-first-training.md`
