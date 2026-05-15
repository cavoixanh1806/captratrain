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
- **Model**: CNN (7 blocks) + BiLSTM (2 layers, hidden=256) ~8.7M params
- **Input**: 128×128 RGB → resize 64×320 (ratio 1:5)
- **Output**: 5 chars from 24-char charset (A,C,D,E,F,H,J,K,L,M,N,P,Q,R,T,U,V,W,X,Y,3,4,7,9)

## Key Files

| File | Purpose |
|------|---------|
| `crnn_model.py` | Model architecture + CTC encode/decode |
| `train_crnn.py` | Training pipeline |
| `eval_crnn.py` | Evaluation metrics |
| `inference_crnn.py` | Inference API |
| `run_all.bat` | Full training workflow |
