# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python (3.10+ per README, observed 3.11.4 in dev env) | `README.md` ("Python 3.10+, RAM 8GB"); `python --version` output |
| Runtime + version | CPython, GPU acceleration via PyTorch + CUDA 12.8 (or CPU fallback) | `requirements.txt` (PyTorch CUDA install instructions); `train_crnn.py` (`torch.cuda.is_available()`) |
| Package manager | `pip` with `venv` virtual environment | `setup.bat` (`python -m venv venv`, `pip install ...`); `requirements.txt` |
| Module/build system | Flat single-package layout, no build system; entry points are CLI scripts and `.bat` orchestrators | repo root listing (all `.py` modules at root); `run_all.bat`, `run_smoke.bat`, `setup.bat` |

### 2) Production Frameworks and Dependencies

`torch` and `torchvision` are installed separately (commented out in `requirements.txt`) so the user picks CUDA vs CPU wheels. The remaining production deps are pinned or floored as listed below.

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| torch | (user-selected: cu128 or cpu wheel) | Tensor ops, CRNN model, CTC loss, AMP, ONNX export | `requirements.txt` (install instructions); `crnn_model.py`, `train_crnn.py`, `inference_crnn.py` |
| torchvision | (user-selected wheel) | `transforms` fallback when albumentations is missing | `requirements.txt`; `dataset_crnn.py` (`from torchvision import transforms`) |
| opencv-python | 4.9.0.80 | Image read/write, resize, color conversion, line/curve drawing for synthetic, remap for wave distortion | `requirements.txt`; `cv2` imported in `dataset_crnn.py`, `inference_crnn.py`, `synthetic_renderer.py`, `eval_crnn.py`, `generate_synthetic_crnn.py` |
| Pillow | 10.3.0 | Font rendering and per-character rotation/blend for synthetic generator | `requirements.txt`; `synthetic_renderer.py` (`PIL.Image`, `ImageDraw`, `ImageFont`); `dataset_crnn.py` (torchvision augmentation fallback) |
| pandas | 2.2.2 | Read/write `metadata.csv` for real and synthetic datasets, train/val split | `requirements.txt`; `dataset_crnn.py`, `eval_crnn.py`, `import_new_data.py`, `self_train.py` |
| numpy | 1.26.4 | Image arrays, normalization, mask manipulation, gradient/noise overlays | `requirements.txt`; imported across all image-handling modules |
| albumentations | 1.4.3 | Strong train-time augmentation (Affine, Perspective, ColorJitter, GaussNoise, Blur, CoarseDropout) with torchvision fallback | `requirements.txt`; `dataset_crnn.py` (`try: import albumentations as A`) |
| onnx | >=1.15 | ONNX export consumer schema (used implicitly by `torch.onnx.export`) | `requirements.txt`; `crnn_model.export_onnx` invokes `torch.onnx.export` |
| onnxruntime | >=1.17 | Runtime for deployed ONNX model (described in README, not imported in code yet) | `requirements.txt`; `README.md` ONNX deploy snippet — [TODO] no source file imports `onnxruntime` |
| psutil | >=5.9.0 | RAM/CPU/freq introspection in system report | `requirements.txt`; `system_info.py` (`import psutil`) |
| tqdm | 4.66.2 | Train and val progress bars (Hugging Face style log dicts) | `requirements.txt`; `train_crnn.py` (`from tqdm import tqdm`) |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| `gh` CLI | Issue tracker and PR operations described in agent docs | `docs/agents/issue-tracker.md` |
| `git` | Version control (history shows iterative pipeline rewrites) | `.git/`, scan output `GIT RECENT COMMITS` |
| `.bat` orchestrators | Windows-only workflow runners (`setup.bat`, `run_smoke.bat`, `run_all.bat`) | repo root |
| Linter / formatter | None configured | scan output: `LINTING AND FORMATTING CONFIG → No linting or formatting config files found` |
| Test runner | None configured | scan output and full repo search; no `pytest.ini`, `tests/`, `test_*.py`, or `conftest.py` |
| CI/CD | None configured | scan output: `CI/CD PIPELINES → No CI/CD pipelines detected` |
| Container runtime | None configured | scan output: `CONTAINERS & ORCHESTRATION → No containerization configs detected` |

### 4) Key Commands

```bash
# Setup (Windows, auto-detects GPU vs CPU)
setup.bat

# Smoke test (~30-60 min CPU, 5 epochs, batch 16, no augment)
run_smoke.bat

# Full training pipeline (import → clean → train 200 epochs → eval)
run_all.bat

# Individual stages
python label_server.py                            # web UI for labeling on http://0.0.0.0:8080
python import_new_data.py                         # copy dataset/*.png → data/, update metadata.csv
python generate_synthetic_crnn.py --count 100000  # synthetic captchas (default 100,000)
python train_crnn.py                              # 200 epochs, real only
python train_crnn.py --use-synthetic --resume     # include synthetic, resume from last checkpoint
python self_train.py --confidence 0.95 --epochs 15
python eval_crnn.py
python inference_crnn.py data/map_00001.png
python system_info.py -f json -o system_info.json
```

### 5) Environment and Config

- Config sources: `requirements.txt` (deps), in-code module-level constants (`DEFAULT_EPOCHS`, `DEFAULT_BATCH_SIZE`, `CHECKPOINT_PATH`, etc.), CLI flags via `argparse`. No `pyproject.toml`, no `setup.cfg`, no `.env`.
- Required env vars: None observed. `[TODO]` confirm none are read at runtime — searched and found no `os.environ` / `os.getenv` reads in source modules.
- Deployment/runtime constraints:
  - Windows-first: `setup.bat`, `run_*.bat`, `system_info.py` Windows PowerShell paths, `train_crnn.py` comment "`num_workers = 0` # Windows: 0 để tránh pickle issue".
  - GPU optional: code uses `cuda if available else cpu`, AMP fp16 enabled only on CUDA.
  - Disk: scan reports 25 MB checkpoint at root (`captcha_crnn_last.pth`); synthetic generator writes 50K-100K PNGs to `data/synthetic_crnn/` (~3-6 GB).

### 6) Evidence

- `requirements.txt`
- `setup.bat`, `run_all.bat`, `run_smoke.bat`
- `crnn_model.py`, `train_crnn.py`, `inference_crnn.py`
- `docs/codebase/.codebase-scan.txt` (sections: STACK DETECTION, LINTING/FORMATTING, CI/CD, CONTAINERS, CODE METRICS)
