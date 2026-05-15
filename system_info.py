"""
system_info.py
===============
Thu thập cấu hình máy + đánh giá khả năng train CRNN+CTC.

Thông tin lấy được:
    - OS, kernel/build
    - CPU (model, cores, threads, max frequency)
    - RAM (total, available, used)
    - Disk (root drive: total, free)
    - GPU (NVIDIA qua nvidia-smi, integrated qua WMI/lspci)
    - Python version + PyTorch CUDA support
    - Đánh giá có thể train không, ước tính thời gian

Format output:
    - "md"   : Markdown bảng (mặc định, đẹp khi mở trong editor)
    - "json" : JSON nested (cho automation)

Usage:
    python system_info.py                      # in ra console (md)
    python system_info.py --format json        # JSON ra console
    python system_info.py -o report.md         # ghi file MD
    python system_info.py -o report.json -f json
    python system_info.py --quiet -o report.json -f json    # silent
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Helpers chung ────────────────────────────────────────────────────────────


def _run(cmd: list[str] | str, timeout: float = 10.0) -> str | None:
    """Chạy 1 command, trả về stdout (str) hoặc None nếu fail.

    Args:
        cmd: command (list hoặc string với shell).
        timeout: giây.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _bytes_to_gb(value: int | float | None) -> float | None:
    """Convert bytes → GB (1024-base)."""
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


# ─── OS ───────────────────────────────────────────────────────────────────────


def collect_os() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }

    # Windows: chi tiết build qua registry/systeminfo
    if info["system"] == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').ProductName"])
        if out:
            info["product_name"] = out
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').DisplayVersion"])
        if out:
            info["display_version"] = out
    elif info["system"] == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["distro"] = line.split("=", 1)[1].strip().strip('"')
                        break
        except OSError:
            pass

    return info


# ─── CPU ──────────────────────────────────────────────────────────────────────


def collect_cpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "architecture": platform.machine(),
    }

    # Cores via os
    try:
        import os
        info["logical_processors"] = os.cpu_count()
    except Exception:
        pass

    # psutil nếu có
    try:
        import psutil  # type: ignore
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_processors"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            info["max_freq_mhz"] = round(freq.max, 0)
            info["current_freq_mhz"] = round(freq.current, 0)
    except ImportError:
        pass

    # Windows: lấy model qua PowerShell
    if platform.system() == "Windows":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1)."
            "Name + '|' + (Get-CimInstance Win32_Processor | Select-Object -First 1)."
            "NumberOfCores + '|' + (Get-CimInstance Win32_Processor | Select-Object -First 1)."
            "NumberOfLogicalProcessors + '|' + (Get-CimInstance Win32_Processor | Select-Object -First 1)."
            "MaxClockSpeed",
        ])
        if out and "|" in out:
            parts = out.split("|")
            if len(parts) >= 4:
                info.setdefault("model", parts[0].strip())
                try:
                    info.setdefault("physical_cores", int(parts[1].strip()))
                    info.setdefault("logical_processors", int(parts[2].strip()))
                    info.setdefault("max_freq_mhz", int(parts[3].strip()))
                except ValueError:
                    pass
    elif platform.system() == "Linux":
        out = _run(["cat", "/proc/cpuinfo"])
        if out:
            for line in out.split("\n"):
                if "model name" in line.lower():
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            info["model"] = out

    info.setdefault("model", platform.processor() or "Unknown")
    return info


# ─── RAM ──────────────────────────────────────────────────────────────────────


def collect_ram() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["total_gb"] = _bytes_to_gb(vm.total)
        info["available_gb"] = _bytes_to_gb(vm.available)
        info["used_gb"] = _bytes_to_gb(vm.used)
        info["used_percent"] = vm.percent
        return info
    except ImportError:
        pass

    # Windows fallback
    if platform.system() == "Windows":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "$m = Get-CimInstance Win32_OperatingSystem; "
            "Write-Output (\"$($m.TotalVisibleMemorySize)|$($m.FreePhysicalMemory)\")",
        ])
        if out and "|" in out:
            try:
                total_kb, free_kb = out.split("|")
                total_b = int(total_kb) * 1024
                free_b = int(free_kb) * 1024
                info["total_gb"] = _bytes_to_gb(total_b)
                info["available_gb"] = _bytes_to_gb(free_b)
                info["used_gb"] = _bytes_to_gb(total_b - free_b)
                info["used_percent"] = round((total_b - free_b) / total_b * 100, 1)
            except (ValueError, ZeroDivisionError):
                pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key, val = parts
                        try:
                            mem[key.strip()] = int(val.strip().split()[0]) * 1024
                        except ValueError:
                            pass
            if "MemTotal" in mem:
                info["total_gb"] = _bytes_to_gb(mem["MemTotal"])
            if "MemAvailable" in mem:
                info["available_gb"] = _bytes_to_gb(mem["MemAvailable"])
            if "MemTotal" in mem and "MemAvailable" in mem:
                used = mem["MemTotal"] - mem["MemAvailable"]
                info["used_gb"] = _bytes_to_gb(used)
                info["used_percent"] = round(used / mem["MemTotal"] * 100, 1)
        except OSError:
            pass

    return info


# ─── Disk ─────────────────────────────────────────────────────────────────────


def collect_disk() -> dict[str, Any]:
    """Lấy thông tin disk của thư mục hiện tại."""
    info: dict[str, Any] = {}
    try:
        usage = shutil.disk_usage(Path.cwd())
        info["path"] = str(Path.cwd().anchor) or str(Path.cwd())
        info["total_gb"] = _bytes_to_gb(usage.total)
        info["free_gb"] = _bytes_to_gb(usage.free)
        info["used_gb"] = _bytes_to_gb(usage.used)
        info["used_percent"] = round(usage.used / usage.total * 100, 1)
    except OSError as e:
        info["error"] = str(e)
    return info


# ─── GPU ──────────────────────────────────────────────────────────────────────


def collect_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "nvidia_smi_available": False,
        "nvidia_devices": [],
        "other_devices": [],
    }

    # Try nvidia-smi
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if out:
        info["nvidia_smi_available"] = True
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                info["nvidia_devices"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(parts[2]),
                    "vram_free_mb": int(parts[3]),
                    "driver_version": parts[4],
                    "compute_capability": parts[5],
                })

    # Other GPUs (integrated etc) via Windows WMI
    if platform.system() == "Windows":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "ForEach-Object { Write-Output (\"$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)\") }",
        ])
        if out:
            for line in out.split("\n"):
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    name, ram, driver = parts[0], parts[1], parts[2]
                    # Skip nếu đã có trong nvidia_devices
                    if any(name in d.get("name", "") for d in info["nvidia_devices"]):
                        continue
                    try:
                        ram_mb = int(ram) // (1024 * 1024) if ram else None
                    except ValueError:
                        ram_mb = None
                    info["other_devices"].append({
                        "name": name.strip(),
                        "vram_mb": ram_mb,
                        "driver_version": driver.strip(),
                    })
    elif platform.system() == "Linux":
        out = _run("lspci | grep -iE 'vga|3d|display'")
        if out:
            for line in out.split("\n"):
                if line.strip():
                    info["other_devices"].append({"name": line.strip()})

    return info


# ─── Python / PyTorch ─────────────────────────────────────────────────────────


def collect_python_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }

    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_total_gb": _bytes_to_gb(
                        torch.cuda.get_device_properties(i).total_memory
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        if hasattr(torch.version, "cuda"):
            info["torch_cuda_compiled"] = torch.version.cuda
    except ImportError:
        info["torch"] = "not installed"

    return info


# ─── Training assessment ──────────────────────────────────────────────────────


def assess_training(report: dict[str, Any]) -> dict[str, Any]:
    """Đánh giá khả năng train CRNN+CTC trên máy này.

    Heuristic dựa trên:
        - Có CUDA GPU NVIDIA không
        - VRAM tối thiểu 4GB cho batch=64
        - RAM tối thiểu 8GB
        - Disk trống ít nhất 5GB cho synthetic
    """
    py = report.get("python", {})
    ram = report.get("ram", {})
    disk = report.get("disk", {})
    gpu = report.get("gpu", {})

    has_cuda = py.get("cuda_available", False)
    nvidia_devices = gpu.get("nvidia_devices", [])
    ram_gb = ram.get("total_gb", 0) or 0
    disk_free_gb = disk.get("free_gb", 0) or 0

    # Tính toán
    issues: list[str] = []
    notes: list[str] = []

    if not has_cuda or not nvidia_devices:
        issues.append("Không có NVIDIA GPU với CUDA → train trên CPU rất chậm (~70 giờ cho 50K samples × 40 epochs).")
    else:
        gpu0 = nvidia_devices[0]
        vram_gb = gpu0["vram_total_mb"] / 1024
        if vram_gb < 4:
            issues.append(f"VRAM {vram_gb:.1f}GB < 4GB. Cần giảm batch xuống 16-32.")
        elif vram_gb < 6:
            notes.append(f"VRAM {vram_gb:.1f}GB đủ cho batch 32. Mặc định 64 cần thử.")
        else:
            notes.append(f"VRAM {vram_gb:.1f}GB OK cho batch 64.")

    if ram_gb < 8:
        issues.append(f"RAM {ram_gb}GB thấp. Pipeline cần ~8GB RAM khi load 50K synthetic.")
    elif ram_gb < 16:
        notes.append(f"RAM {ram_gb}GB đủ. Có thể cần đóng app khác khi train.")

    if disk_free_gb < 5:
        issues.append(f"Disk free {disk_free_gb}GB < 5GB. 50K synthetic chiếm ~3GB.")

    # Verdict
    if not issues:
        verdict = "READY"
        time_estimate = "~30-45 phút trên RTX 3060+, ~10-15 phút trên RTX 5090"
    elif has_cuda and nvidia_devices:
        verdict = "MARGINAL"
        time_estimate = "Có thể train, cần giảm batch size"
    elif ram_gb >= 8 and disk_free_gb >= 3:
        verdict = "CPU_ONLY"
        time_estimate = (
            "~10-15h với pipeline mặc định (754 real × 200 epochs). "
            "Khuyến nghị smoke-test với --epochs 5 → ~1-2h"
        )
    else:
        verdict = "NOT_RECOMMENDED"
        time_estimate = "Máy không đủ tài nguyên."

    return {
        "verdict": verdict,
        "issues": issues,
        "notes": notes,
        "estimated_time": time_estimate,
        "recommended_action": _recommend_action(verdict),
    }


def _recommend_action(verdict: str) -> str:
    return {
        "READY": "Chạy `run_all.bat` (train + eval). Nếu accuracy < 90%, bật synthetic: `python generate_synthetic_crnn.py && python train_crnn.py --use-synthetic`.",
        "MARGINAL": "Chạy với batch giảm: `python train_crnn.py --batch-size 32`.",
        "CPU_ONLY": "Push code lên git → train trên máy GPU. Trên máy này chỉ smoke test: `python train_crnn.py --epochs 5 --batch-size 16` (~1-2h).",
        "NOT_RECOMMENDED": "Nâng cấp hardware (NVIDIA GPU + 16GB RAM) hoặc dùng cloud (Colab, Kaggle, Lambda).",
    }.get(verdict, "")


# ─── Build full report ────────────────────────────────────────────────────────


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": collect_os(),
        "cpu": collect_cpu(),
        "ram": collect_ram(),
        "disk": collect_disk(),
        "gpu": collect_gpu(),
        "python": collect_python_env(),
    }
    report["training_assessment"] = assess_training(report)
    return report


# ─── Format: JSON ─────────────────────────────────────────────────────────────


def format_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


# ─── Format: Markdown ─────────────────────────────────────────────────────────


def _md_kv_table(d: dict[str, Any], skip: set[str] | None = None) -> str:
    """Format dict thành bảng MD 2 cột."""
    skip = skip or set()
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# System Information Report")
    lines.append("")
    lines.append(f"**Generated:** {report['timestamp']}")
    lines.append(f"**Hostname:** `{report['hostname']}`")
    lines.append("")

    # OS
    lines.append("## Operating System")
    lines.append("")
    lines.append(_md_kv_table(report["os"]))
    lines.append("")

    # CPU
    lines.append("## CPU")
    lines.append("")
    lines.append(_md_kv_table(report["cpu"]))
    lines.append("")

    # RAM
    lines.append("## RAM")
    lines.append("")
    lines.append(_md_kv_table(report["ram"]))
    lines.append("")

    # Disk
    lines.append("## Disk (current drive)")
    lines.append("")
    lines.append(_md_kv_table(report["disk"]))
    lines.append("")

    # GPU
    lines.append("## GPU")
    lines.append("")
    gpu = report["gpu"]
    lines.append(f"**nvidia-smi available:** `{gpu['nvidia_smi_available']}`")
    lines.append("")
    if gpu["nvidia_devices"]:
        lines.append("### NVIDIA devices")
        lines.append("")
        lines.append("| Index | Name | VRAM Total (MB) | VRAM Free (MB) | Driver | Compute |")
        lines.append("|---|---|---|---|---|---|")
        for d in gpu["nvidia_devices"]:
            lines.append(
                f"| {d['index']} | {d['name']} | {d['vram_total_mb']} | "
                f"{d['vram_free_mb']} | {d['driver_version']} | {d['compute_capability']} |"
            )
        lines.append("")
    else:
        lines.append("_Không có NVIDIA GPU._")
        lines.append("")
    if gpu["other_devices"]:
        lines.append("### Other devices (integrated, etc.)")
        lines.append("")
        lines.append("| Name | VRAM (MB) | Driver |")
        lines.append("|---|---|---|")
        for d in gpu["other_devices"]:
            lines.append(
                f"| {d.get('name', '')} | {d.get('vram_mb', '-')} | "
                f"{d.get('driver_version', '-')} |"
            )
        lines.append("")

    # Python
    lines.append("## Python Environment")
    lines.append("")
    lines.append(_md_kv_table(report["python"], skip={"cuda_devices"}))
    py = report["python"]
    if py.get("cuda_devices"):
        lines.append("")
        lines.append("### PyTorch CUDA devices")
        lines.append("")
        lines.append("| Index | Name | VRAM Total (GB) |")
        lines.append("|---|---|---|")
        for d in py["cuda_devices"]:
            lines.append(f"| {d['index']} | {d['name']} | {d['vram_total_gb']} |")
        lines.append("")

    # Assessment
    lines.append("## Training Assessment")
    lines.append("")
    a = report["training_assessment"]
    lines.append(f"**Verdict:** `{a['verdict']}`")
    lines.append("")
    lines.append(f"**Estimated time:** {a['estimated_time']}")
    lines.append("")
    if a["issues"]:
        lines.append("### Issues")
        lines.append("")
        for issue in a["issues"]:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")
    if a["notes"]:
        lines.append("### Notes")
        lines.append("")
        for note in a["notes"]:
            lines.append(f"- ℹ️ {note}")
        lines.append("")
    lines.append("### Recommended action")
    lines.append("")
    lines.append(f"> {a['recommended_action']}")
    lines.append("")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thu thập cấu hình máy và đánh giá khả năng train CRNN+CTC.",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["md", "json"],
        default="md",
        help="Output format (default: md)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path. Nếu không có thì in ra stdout.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Không in ra stdout khi đã ghi file.",
    )
    args = parser.parse_args()

    report = build_report()

    if args.format == "json":
        content = format_json(report)
    else:
        content = format_markdown(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        if not args.quiet:
            print(f"[OK] Report saved to: {args.output.resolve()}")
            verdict = report["training_assessment"]["verdict"]
            print(f"[INFO] Verdict: {verdict}")
    else:
        print(content)


if __name__ == "__main__":
    main()
