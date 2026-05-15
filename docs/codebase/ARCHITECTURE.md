# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: Linear ML training/inference pipeline organized by stage (data ingestion → augmentation → model → loss → checkpoint → evaluation → ONNX export). Modules sit side by side at the repo root and are wired together by direct imports and `.bat` orchestrators rather than a framework.
- Why this classification: There is no web framework, no IoC container, no plugin system, and no `src/` layering. Each `.py` module owns a single pipeline stage and is invoked either by another module's import or by an orchestrator script. Evidence: flat module list at repo root; `run_all.bat` calls `import_new_data.py → train_crnn.py → eval_crnn.py` in sequence.
- Primary constraints (drawn from intent docs and code):
  - Fixed-charset (24 chars), fixed-length (5 chars) Minecraft Map CAPTCHA — drives charset, model output size, and `_enforce_length` post-processing.
  - Small real corpus (754 labelled images) — motivates synthetic augmentation, calibrated rendering, and self-training (`docs/adr/0002-synthetic-first-training.md`).
  - Windows + optional NVIDIA GPU target — motivates `.bat` scripts, `num_workers=0` choice, AMP only on CUDA, and the `system_info.py` verdict tool.

### 2) System Flow

```text
dataset/*.png  ─►  import_new_data.py  ─►  data/*.png + data/metadata.csv
                                                  │
                                                  ▼
synthetic_renderer.render_text_on_image  ─►  generate_synthetic_crnn.py  ─►  data/synthetic_crnn/*.png + metadata.csv
                                                  │
                                                  ▼
                            dataset_crnn.create_crnn_datasets  (real train/val split + optional synthetic)
                                                  │
                            (DataLoader + collate_fn for variable-length CTC labels)
                                                  ▼
                            train_crnn.main                                      
                              CRNN forward → log_softmax → CTCLoss → backward    
                              AdamW (warmup→cosine) + AMP fp16 + grad clip 5.0   
                              save best (val_exact_match) + last (resume)        
                                                  │
                                                  ▼
                                  captcha_crnn_model.pth + .onnx
                                                  │
                            ┌─────────────────────┼─────────────────────┐
                            ▼                     ▼                     ▼
                self_train.py            eval_crnn.py            inference_crnn.CRNNCaptchaSolver
                (pseudo-label round 2)   (metrics + verdict)     (single/batch/with-confidence)
```

Detailed steps with file evidence:

1. New labelled images land in `dataset/map_<LABEL>.png`. `import_new_data.py:main` validates the 5-char label, dedupes by label against existing `metadata.csv`, and copies into `data/map_<NNNNN>.png` with sequential numbering.
2. `generate_synthetic_crnn.generate` calls `synthetic_renderer.render_text_on_image` for each random label and writes the rendered PNG plus a row in `data/synthetic_crnn/metadata.csv`.
3. `dataset_crnn.create_crnn_datasets` reads `data/metadata.csv`, splits 85/15 train/val (`val_split=0.15`, `seed=42`), optionally concatenates synthetic into the train side via `torch.utils.data.ConcatDataset`. Real-only validation preserves domain.
4. `dataset_crnn.collate_fn` produces a dict with concatenated CTC labels and per-sample `label_lengths` (sum-of-lengths layout required by `nn.CTCLoss`).
5. `train_crnn.train_one_epoch` runs the forward pass through `CRNN`, applies `log_softmax`, computes `CTCLoss` with `input_lengths = T` for every sample, backprops under `torch.amp.GradScaler` on CUDA, clips gradients at 5.0, and steps the warmup-cosine scheduler every batch. `train_crnn.validate` decodes with `decode_greedy` and computes exact-match plus CER. The best checkpoint is saved by `val_exact_match`; a `last` checkpoint is saved every epoch for `--resume`.
6. After training, `crnn_model.export_onnx` produces `captcha_crnn_model.onnx` with dynamic batch axis.
7. `eval_crnn.evaluate` reuses `CRNNCaptchaSolver.solve_batch_with_confidence` over all real samples and prints exact match, CER, per-position accuracy, top-10 confusions, low-confidence wrongs, and a verdict against the 90% target.
8. `self_train.select_high_confidence_samples` predicts every real sample, keeps only `conf ≥ threshold AND pred == label`, then `fine_tune_round2` resumes training at LR 1e-4 with synthetic as a 10x regularizer subset and saves any improved checkpoint.

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `crnn_model.py` | Network architecture, charset/index maps, CTC encode/greedy decode, save/load, ONNX export | Disk I/O for datasets, training loop | `crnn_model.py` (no `pandas`/`cv2.imread` use) |
| `dataset_crnn.py` | `Dataset` impls, image read+normalize, augmentation, train/val factory, collate | Model definition, optimizer | `dataset_crnn.py` (imports model only for charset and shape) |
| `synthetic_renderer.py` + `generate_synthetic_crnn.py` | Pixel-level rendering and bulk generation to disk | Model or training logic | both modules (no `torch` import) |
| `train_crnn.py` | Training loop, scheduler, checkpointing, ONNX export trigger | Architecture changes | `train_crnn.py` |
| `self_train.py` | Pseudo-label selection, round-2 fine-tune, re-export ONNX | Round-1 training behaviour redefinition | `self_train.py` (reuses `train_crnn.train_one_epoch`/`validate`) |
| `inference_crnn.py` | Solver API, batched preprocessing, length enforcement | Training | `inference_crnn.py` |
| `eval_crnn.py` | Aggregate metrics and verdict | Training, model surgery | `eval_crnn.py` (delegates to solver) |
| `import_new_data.py` + `label_server.py` | Data acquisition and labelling UX | Model or training logic | both modules |
| `system_info.py` | Host introspection and training-readiness verdict | Project-specific training decisions | `system_info.py` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Module-level constants for cross-module config | `crnn_model.INPUT_HEIGHT`, `INPUT_WIDTH`, `NUM_CLASSES`, `CTC_BLANK_INDEX`; `train_crnn.CHECKPOINT_PATH`, `LAST_CHECKPOINT_PATH`, `ONNX_PATH` | Single source of truth shared across data, train, and inference modules |
| Smoke test in `__main__` | `crnn_model.py`, `dataset_crnn.py`, `synthetic_renderer.py` | Run-as-script self-check before plumbing into pipeline |
| Optional dependency with fallback | `dataset_crnn` (`try: import albumentations` → torchvision transforms); `system_info` (`try: import psutil`); `system_info` (`try: import torch`) | Code degrades gracefully on minimal installs |
| Dict batches for loss inputs | `dataset_crnn.collate_fn` returns `{"images", "labels", "label_lengths", "texts"}` | Hugging Face-Trainer-shaped contract used by `train_crnn`/`self_train`/`validate` |
| Two-checkpoint strategy | `train_crnn` saves `captcha_crnn_model.pth` (best by val_exact_match) and `captcha_crnn_last.pth` (every epoch, for `--resume`) | Decouple "best for inference" from "training state for resume" |
| Greedy CTC decode + length enforcement | `crnn_model.decode_greedy` then `inference_crnn._enforce_length` | Guarantees a 5-char output even when CTC predicts 4 or 6; deliberate trade-off (always-wrong fallback char) |
| Subprocess + powershell helper for host introspection | `system_info._run` | Cross-tool collection on Windows without an extra dep |

### 5) Known Architectural Risks

- Doc/code drift on hyperparameters: Resolved (2026-05-15): canonical `hidden_size=128`, `DEFAULT_EPOCHS=200`; docs synced. `crnn_model.CRNN.__init__` is the single source of truth for `hidden_size`; `train_crnn.DEFAULT_EPOCHS` is the single source of truth for the default epoch count. `README.md`, `PIPELINE_SUMMARY.md`, `CLAUDE.md`, and `docs/adr/0001-crnn-ctc-over-softmax.md` now reference the canonical constants by name and quote `count_parameters() == 2,186,553`.
- High-churn names in git history (`train.py`, `inference.py`, `dataset.py`, `generate_unet_data.py`, `train_unet.py`) no longer exist; the recent commit `bf743f3` "Replace TrOCR with CRNN+CTC pipeline" was a full rewrite. Anyone reading the README "So sánh với repo tham khảo" or the high-churn list should know the U-Net/TrOCR variants are gone.
- `inference_crnn._enforce_length` pads short predictions with `CAPTCHA_CHARSET[0]` (`'A'`). This guarantees length=5 but also guarantees a wrong character whenever the model output is short. CER reported by `eval_crnn` will be biased upward in those cases (worse than a length-aware decoder).
- `data/_crnn_train.csv` and `data/_crnn_val.csv` are written by `dataset_crnn.create_crnn_datasets` then immediately `unlink`ed in the same call after dataset construction. Two simultaneous training runs in the same working directory race on these files.
- `import_new_data.py` dedupes incoming images by label only (`existing_labels` set). Two distinct real captchas that happen to have the same label silently lose the newer one.
- `label_server.py` binds to `0.0.0.0:8080` with no auth. Acceptable on a private LAN; risky on a shared network.

### 6) Evidence

- `crnn_model.py`, `dataset_crnn.py`, `train_crnn.py`, `self_train.py`, `inference_crnn.py`, `eval_crnn.py`
- `docs/adr/0001-crnn-ctc-over-softmax.md`, `docs/adr/0002-synthetic-first-training.md`
- `run_all.bat`, `run_smoke.bat`
- `docs/codebase/.codebase-scan.txt` (GIT RECENT COMMITS, HIGH-CHURN FILES)
