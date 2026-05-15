# CLAUDE.md

Agent instructions for the CAPTCHA Solver (CRNN+CTC) project.

## Agent skills

### Issue tracker

Issues live in GitHub (`cavoixanh1806/captratrain`). Use `gh` CLI for all operations. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. Read `CONTEXT.md` at root + `docs/adr/` for architectural decisions. See `docs/agents/domain.md`.

## Project Overview

This is a CAPTCHA solver for Minecraft Map CAPTCHA using CRNN+CTC architecture.
- **Goal**: exact_match ≥ 90%, CER ≤ 10%
- **Model**: CNN (7 blocks) + BiLSTM (2 layers, hidden=128 (see `crnn_model.CRNN` default)) ~2.18M params (`count_parameters() == 2,186,553`)
- **Input**: 128×128 RGB → resize 64×320 (ratio 1:5)
- **Output**: 5 chars from 24-char charset (A,C,D,E,F,H,J,K,L,M,N,P,Q,R,T,U,V,W,X,Y,3,4,7,9)

## Canonical hyperparams (single source of truth)

The values below are defined in source code; docs reference them by name to avoid drift.

| Param | Value | Source |
|-------|-------|--------|
| `hidden_size` | 128 | `crnn_model.CRNN.__init__` default |
| `count_parameters()` | 2,186,553 | `crnn_model.CRNN(...).count_parameters()` |
| `DEFAULT_EPOCHS` | 200 | `train_crnn.DEFAULT_EPOCHS` |
| `DEFAULT_BATCH_SIZE` | 32 | `train_crnn.DEFAULT_BATCH_SIZE` |
| `DEFAULT_LR` | 5e-4 | `train_crnn.DEFAULT_LR` |
| `WARMUP_STEPS` | `max(200, steps_per_epoch * 2)` | `train_crnn` (≥ 2 epochs warmup floor) |
| `INPUT_HEIGHT × INPUT_WIDTH` | 64 × 320 | `crnn_model` constants |
| `NUM_CLASSES` / `CTC_BLANK_INDEX` | 25 / 0 | `crnn_model` constants |

DataLoader `num_workers` follows an auto policy: on Windows `min(4, cpu_count // 2)`, on Linux/macOS `min(8, cpu_count // 2)`, falling back to `0` on the torchvision path or when the user passes `--num-workers 0`. When `num_workers > 0` the loader sets `persistent_workers=True` and `prefetch_factor=4`.

## Key Files

| File | Purpose |
|------|---------|
| `crnn_model.py` | Model architecture + CTC encode/decode |
| `train_crnn.py` | Training pipeline |
| `eval_crnn.py` | Evaluation metrics |
| `inference_crnn.py` | Inference API |
| `run_all.bat` | Full training workflow |
