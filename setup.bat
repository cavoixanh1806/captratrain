@echo off
REM ============================================================================
REM setup.bat — Tu dong setup moi truong tu dau cho may train moi
REM ============================================================================
REM Quy trinh:
REM   1. Kiem tra Python 3.10+
REM   2. Tao venv neu chua co
REM   3. Activate venv
REM   4. Upgrade pip
REM   5. Cai PyTorch (CUDA neu co GPU NVIDIA, fallback CPU)
REM   6. Cai requirements.txt
REM   7. Verify imports + check system_info
REM
REM Su dung:
REM   setup.bat              (auto-detect GPU)
REM   setup.bat --cpu        (force CPU)
REM   setup.bat --cuda 121   (force CUDA 12.1, vd RTX 30xx tren driver cu)
REM   setup.bat --cuda 128   (CUDA 12.8, RTX 40xx/50xx)
REM ============================================================================

setlocal enabledelayedexpansion

REM === Parse args ===
set FORCE_MODE=auto
set CUDA_VER=128
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--cpu" (
    set FORCE_MODE=cpu
    shift & goto parse_args
)
if /i "%~1"=="--cuda" (
    set FORCE_MODE=cuda
    set CUDA_VER=%~2
    shift & shift & goto parse_args
)
echo [WARN] Khong nhan dien arg: %~1
shift & goto parse_args
:end_parse

echo ============================================================
echo CRNN+CTC CAPTCHA - SETUP MOI TRUONG
echo ============================================================
echo.

REM === Buoc 1: Check Python ===
echo [1/7] Kiem tra Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Khong tim thay Python tren PATH.
    echo         Tai Python 3.10+ tu https://www.python.org/downloads/
    echo         Khi cai dat, NHO check "Add Python to PATH".
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] %PYVER%
echo.

REM === Buoc 2: Tao venv ===
echo [2/7] Tao venv neu chua co...
if exist venv (
    echo [INFO] venv da ton tai, bo qua.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Khong tao duoc venv. Kiem tra Python install.
        pause
        exit /b 1
    )
    echo [OK] Da tao venv\
)
echo.

REM === Buoc 3: Activate venv ===
echo [3/7] Activate venv...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Khong activate duoc venv.
    pause
    exit /b 1
)
echo [OK] Activated. Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.

REM === Buoc 4: Upgrade pip ===
echo [4/7] Upgrade pip...
python -m pip install --upgrade pip wheel setuptools --quiet
if errorlevel 1 (
    echo [WARN] pip upgrade fail, co the tiep tuc duoc.
)
echo [OK] pip da upgrade.
echo.

REM === Buoc 5: Detect GPU + cai PyTorch ===
echo [5/7] Cai PyTorch...
set PYTORCH_INDEX=
if "%FORCE_MODE%"=="cpu" (
    set PYTORCH_INDEX=--index-url https://download.pytorch.org/whl/cpu
    echo [INFO] Force CPU mode.
) else if "%FORCE_MODE%"=="cuda" (
    set PYTORCH_INDEX=--index-url https://download.pytorch.org/whl/cu%CUDA_VER%
    echo [INFO] Force CUDA %CUDA_VER%.
) else (
    REM Auto-detect: thu nvidia-smi
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Khong co nvidia-smi. Cai PyTorch CPU.
        set PYTORCH_INDEX=--index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo [INFO] Phat hien NVIDIA GPU. Cai PyTorch CUDA 12.8.
        set PYTORCH_INDEX=--index-url https://download.pytorch.org/whl/cu128
    )
)

echo [INFO] Lenh: pip install torch torchvision %PYTORCH_INDEX%
pip install torch torchvision %PYTORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] Cai PyTorch fail. Thu lai voi --cpu hoac --cuda 121.
    pause
    exit /b 1
)
echo [OK] PyTorch cai xong.
echo.

REM === Buoc 6: Cai requirements ===
echo [6/7] Cai requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Cai requirements fail.
    pause
    exit /b 1
)
echo [OK] Requirements cai xong.
echo.

REM === Buoc 7: Verify ===
echo [7/7] Verify imports + check system...
python -c "import torch, cv2, numpy, pandas, albumentations, onnx; print('[OK] Tat ca modules import duoc')"
if errorlevel 1 (
    echo [ERROR] Co module loi import.
    pause
    exit /b 1
)
echo.

python -c "import torch; print('[INFO] PyTorch:', torch.__version__); print('[INFO] CUDA available:', torch.cuda.is_available()); print('[INFO] Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 'CPU only')"
echo.

echo ============================================================
echo Verify import cac module project...
echo ============================================================
python -c "import crnn_model, dataset_crnn, train_crnn, inference_crnn, eval_crnn, generate_synthetic_crnn, synthetic_renderer, system_info, self_train; print('[OK] All 9 project modules import OK')"
if errorlevel 1 (
    echo [ERROR] Co module project loi import.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo System info
echo ============================================================
python system_info.py

echo.
echo ============================================================
echo [DONE] Setup hoan tat!
echo ============================================================
echo.
echo Buoc tiep theo:
echo   1. Smoke test (~5-10 phut tren GPU): run_smoke.bat
echo   2. Train chinh (~30-45 phut tren RTX 3060): run_all.bat
echo.
echo Xem PIPELINE_SUMMARY.md de biet chi tiet.
pause
