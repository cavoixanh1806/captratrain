# External Integrations

## Core Sections (Required)

This project is a self-contained, on-host machine learning pipeline. It has no third-party APIs, no databases, no message queues, no cloud SDKs, and no telemetry pipeline. All "integrations" below are local: filesystem artifacts, a local labeling HTTP server, and the host-introspection helpers in `system_info.py`.

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| Local filesystem (`data/`, `dataset/`, `data/synthetic_crnn/`) | Filesystem | Image storage, `metadata.csv`, checkpoints, ONNX export | OS file permissions | high | `dataset_crnn.py`, `import_new_data.py`, `train_crnn.py` (`CHECKPOINT_PATH`, `LAST_CHECKPOINT_PATH`, `ONNX_PATH`) |
| `label_server.py` HTTP server | HTTP API (local) | Web labeling UI; `GET /`, `GET /api/metadata`, `GET /images/{name}`, `POST /api/save` | None (anonymous) | medium (only when actively labelling) | `label_server.py` (`HOST = "0.0.0.0"`, `PORT = 8080`, `LabelHandler.do_GET/do_POST`) |
| `nvidia-smi` (subprocess) | CLI tool | Detect NVIDIA GPUs, VRAM, driver, compute capability for the readiness verdict | Local execution | low (diagnostic only) | `system_info.collect_gpu` (`_run(["nvidia-smi", "--query-gpu=...", ...])`) |
| Windows PowerShell (subprocess) | CLI tool | Read OS, CPU, RAM, GPU details via `Get-CimInstance` and registry on Windows | Local execution | low (diagnostic only) | `system_info.collect_os/collect_cpu/collect_ram/collect_gpu` |
| `pip` package index (`download.pytorch.org/whl/cu128` and `cpu`) | Package registry | Install PyTorch wheels during setup | Anonymous public access | high (one-shot at install time) | `requirements.txt` install instructions; `setup.bat` |
| GitHub (via `gh` CLI) | External API | Issue tracker (out-of-band, agent docs) | `gh auth` (developer credential) | medium (workflow only, not runtime) | `docs/agents/issue-tracker.md` |
| `ddddocr` / `onnxruntime` (described in README) | Inference runtime | Suggested deployment runtime for the exported ONNX model | n/a | external (deployment only) | `README.md` "Deploy với ONNX runtime" section. `[TODO]` no source module imports `onnxruntime`; deployment is left to consumers. |

No outbound HTTP/JSON APIs, no DB drivers, no message queues, no cloud SDKs are imported anywhere in source. Verified by repo-wide search for `requests.`, `urllib.request`, `psycopg`, `pymysql`, `pymongo`, `sqlite3`, `sqlalchemy`, `boto3`, `socket.` (only `urllib.parse` is used in `label_server.py`).

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| `data/metadata.csv` | Single source of truth mapping `filename → text` for the 754 real images | `pandas.read_csv` / `to_csv` in `dataset_crnn.py`, `import_new_data.py`, `eval_crnn.py`, `self_train.py`; raw `csv.DictReader/DictWriter` in `label_server.py` | Concurrent writes (label server + import + training temp split) can race; no schema validation beyond 5-char length check | `import_new_data.py:METADATA_CSV`, `dataset_crnn.create_crnn_datasets`, `label_server.read_metadata/write_metadata` |
| `data/synthetic_crnn/metadata.csv` | Filename → label for generated synthetic images | `pandas.read_csv` in `dataset_crnn.py`; `csv.DictWriter` in `generate_synthetic_crnn.py` | Regenerated each run (`shutil.rmtree(output_dir)` if exists). Disk usage 3-6 GB at 50K-100K images. | `generate_synthetic_crnn.generate` |
| `captcha_crnn_model.pth` | "Best by val_exact_match" checkpoint payload (state_dict + charset metadata) | `torch.save/torch.load` via `crnn_model.save_crnn`/`load_crnn` | Loss of file forces full retrain; ~25 MB | `crnn_model.save_crnn`, `train_crnn.py` |
| `captcha_crnn_last.pth` | Per-epoch resume state (state_dict, optimizer, scheduler, scaler, epoch, best_val_em) | `torch.save/torch.load` directly in `train_crnn.py` | Used by `--resume`; not loaded by `inference_crnn` | `train_crnn.py` |
| `captcha_crnn_model.onnx` | Deployment artifact via `torch.onnx.export` (opset 14, dynamic batch) | `crnn_model.export_onnx` | Re-exported from best checkpoint at end of training and after self-train | `crnn_model.export_onnx` |
| `data/_crnn_train.csv`, `data/_crnn_val.csv` | Transient temp files used by `create_crnn_datasets` to materialise the 85/15 split | Written, read into memory, then `unlink(missing_ok=True)` in the same function call | Race between two parallel training runs in the same directory | `dataset_crnn.create_crnn_datasets` |
| `data/_high_confidence.csv` | Filtered list of high-confidence + ground-truth-matching predictions used as round-2 training set | `pandas.DataFrame.to_csv` | Overwritten each `self_train.py` run | `self_train.HIGH_CONF_OUTPUT_CSV`, `self_train.select_high_confidence_samples` |
| `train_log.txt` / `smoke_log.txt` | UTF-8 run logs written by the `.bat` orchestrators via an inline `python -c` tee | Append-mode file writes from a Python tee subprocess | Not rotated; can grow large over many runs | `run_all.bat`, `run_smoke.bat` |
| `system_info.md` / `system_info.json` | Optional host report output | `Path.write_text` | Listed in `.gitignore` because it is host-specific | `system_info.format_markdown/format_json`, `.gitignore` |

No relational DB, no key-value store, no object storage.

### 3) Secrets and Credentials Handling

- Credential sources: none. There are no API keys, no `.env` files, no secret managers, and no `os.environ`/`os.getenv` reads anywhere in the source modules. (Verified by repo-wide search across `*.py`.)
- Hardcoding checks: only one hardcoded path is present — `label_server.DATA_DIR = Path(r"C:\Users\Administrator\Desktop\captratrain\data")`. This is a developer-machine path, not a credential, but it makes `label_server.py` non-portable.
- Rotation or lifecycle notes: not applicable — no secrets.

### 4) Reliability and Failure Behavior

- Retry/backoff behavior: none. There are no network calls to retry. `system_info._run` enforces a 10-second timeout per subprocess and returns `None` on `TimeoutExpired`/`FileNotFoundError`/`OSError`, but does not retry.
- Timeout policy: configured only in `system_info._run(timeout=10.0)`. `label_server.py` uses Python's stdlib `HTTPServer` defaults (no timeouts).
- Circuit-breaker or fallback behavior:
  - Optional dependency fallback: `dataset_crnn._build_albu_aug` falls back to `torchvision.transforms.Compose` when `albumentations` is unavailable; `system_info` falls back from `psutil` to platform-specific subprocess calls.
  - ONNX export failure is non-fatal: `train_crnn.main` and `self_train.fine_tune_round2` wrap `export_onnx` in `try/except` and log a warning instead of aborting.
  - Length enforcement: `inference_crnn._enforce_length` truncates predictions longer than 5 chars and pads short predictions with `CAPTCHA_CHARSET[0]` to keep downstream contracts intact (at the cost of a guaranteed-wrong char on short outputs).

### 5) Observability for Integrations

- Logging around external calls: `nvidia-smi`/PowerShell calls in `system_info` produce structured info via `logging` only on errors (the helper itself silently returns `None`). The HTTP server filters access logs to only `/api/*` paths via `LabelHandler.log_message`.
- Metrics/tracing coverage: none. No Prometheus, OpenTelemetry, StatsD, or APM instrumentation. Training emits Hugging Face style log dicts (`{loss, grad_norm, learning_rate, epoch}` and `{eval_loss, eval_cer, eval_exact_match, eval_runtime, ...}`) printed to stdout and captured by the orchestrator's tee into `train_log.txt`/`smoke_log.txt`.
- Missing visibility gaps:
  - No correlation/ID for the labeling HTTP server (each `POST /api/save` is independent).
  - No structured event for "best checkpoint replaced" — only an INFO log line.
  - No metric for synthetic generation throughput beyond the periodic `logger.info(... 5.0%)` ticks.

### 6) Evidence

- `label_server.py`, `system_info.py`, `train_crnn.py`, `self_train.py`, `crnn_model.py`, `dataset_crnn.py`, `inference_crnn.py`, `generate_synthetic_crnn.py`, `import_new_data.py`
- `requirements.txt`, `setup.bat`, `run_all.bat`, `run_smoke.bat`
- `.gitignore` (artifact lifecycle)
- `docs/codebase/.codebase-scan.txt` (sections: ENVIRONMENT VARIABLE TEMPLATES, SECURITY & COMPLIANCE)
- `docs/agents/issue-tracker.md`
