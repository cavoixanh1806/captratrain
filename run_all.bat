@echo off
REM ============================================================================
REM run_all.bat — CRNN+CTC pipeline (full training + eval, resume-aware)
REM ============================================================================
REM Workflow:
REM   0. Import anh moi tu dataset/ neu co
REM   1. Train CRNN+CTC tren real (`data/metadata.csv`, val 15%, augment ON)
REM      - Checkpoint sau MOI epoch: captcha_crnn_last.pth (resume)
REM      - Best model theo val_exact_match: captcha_crnn_model.pth (+ .onnx)
REM   2. Eval tren toan bo real
REM   3. Archive artifact (model, onnx, log) vao runs\run_<timestamp>\
REM
REM Su dung:
REM   run_all.bat                                  Clean train, default config
REM   run_all.bat --resume                         Resume tu captcha_crnn_last.pth
REM   run_all.bat --epochs 100                     Override epoch
REM   run_all.bat --batch-size 64 --num-workers 4  Override DataLoader
REM   run_all.bat --resume --epochs 50             Resume + chinh epoch budget
REM
REM Output thu muc rieng:
REM   runs\run_<YYYYMMDD_HHMMSS>\
REM     ├ train_log.txt
REM     ├ metrics.csv               (per-epoch: loss, eval_loss, eval_em, gap, ...)
REM     ├ eval_summary.json         (exact_match, CER, confusion_matrix, ...)
REM     ├ captcha_crnn_model.pth   (best)
REM     ├ captcha_crnn_last.pth    (last)
REM     └ captcha_crnn_model.onnx
REM
REM TAM THOI KHONG bat:
REM   - Synthetic (chay rieng: python generate_synthetic_crnn.py)
REM   - Self-train (chay rieng: python self_train.py)
REM
REM Thoi gian: ~30-45 phut RTX 3060 / 200 epoch / ~500 anh real
REM ============================================================================

setlocal enabledelayedexpansion

REM === Parse args (truyen thang xuong train_crnn.py, ngoai tru --resume duoc xu ly o day) ===
set RESUME=0
set TRAIN_ARGS=
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--resume" (
    set RESUME=1
    set TRAIN_ARGS=!TRAIN_ARGS! --resume
    shift & goto parse_args
)
REM Forward all other args (--epochs, --batch-size, --lr, --num-workers, --no-augment, --use-synthetic, --no-real)
set TRAIN_ARGS=!TRAIN_ARGS! %~1
shift & goto parse_args
:end_parse

REM === Auto-detect Python (uu tien venv local) ===
set PY=python
if exist "venv\Scripts\python.exe" (
    set PY=venv\Scripts\python.exe
    echo [INFO] Su dung venv: %PY%
) else (
    echo [WARN] Khong tim thay venv\Scripts\python.exe, fallback ve: %PY%
)

REM === Tao thu muc runs\ neu chua co ===
if not exist runs mkdir runs

REM === Timestamp + thu muc rieng cho run nay ===
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value ^| find "="') do set DT=%%I
set TS=%DT:~0,8%_%DT:~8,6%
set RUN_DIR=runs\run_%TS%
mkdir "%RUN_DIR%"

set LOGFILE=%RUN_DIR%\train_log.txt
echo ============================================================ > "%LOGFILE%"
echo CRNN+CTC CAPTCHA - TRAINING WORKFLOW >> "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo Python:  %PY% >> "%LOGFILE%"
echo Resume:  %RESUME% >> "%LOGFILE%"
echo Args:    %TRAIN_ARGS% >> "%LOGFILE%"
echo Run dir: %RUN_DIR% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

REM === Cung mirror sang train_log.txt o root cho tools ke thua doc (test_property1, ...) ===
copy /y "%LOGFILE%" train_log.txt >nul 2>&1

echo ============================================================
echo CRNN+CTC CAPTCHA - TRAINING WORKFLOW
echo ============================================================
echo Run dir:  %RUN_DIR%
echo Log file: %LOGFILE%
echo Resume:   %RESUME%
echo Args:    %TRAIN_ARGS%
echo.

REM === Buoc 0: Import data moi tu dataset/ ===
if exist dataset (
    echo BUOC 0/3: Import data moi tu dataset\
    %PY% import_new_data.py 2>&1 | %PY% -c "import sys; f=open(r'%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
    if errorlevel 1 goto error
    echo [OK] Import xong.
    echo.
)

REM === Dem so anh real ===
set /a IMG_COUNT=0
for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
echo [INFO] Co %IMG_COUNT% anh real trong data/
echo [INFO] Co %IMG_COUNT% anh real trong data/ >> "%LOGFILE%"

if %IMG_COUNT% LSS 50 (
    echo [ERROR] Can it nhat 50 anh real de train. Hien chi co %IMG_COUNT%.
    goto error
)

REM === Clean train: xoa model cu. Resume train: GIU NGUYEN captcha_crnn_last.pth ===
if "%RESUME%"=="0" (
    echo [CLEAN] Xoa model cu de clean train...
    echo [CLEAN] Xoa model cu de clean train... >> "%LOGFILE%"
    if exist captcha_crnn_model.pth del /q captcha_crnn_model.pth
    if exist captcha_crnn_model.onnx del /q captcha_crnn_model.onnx
    if exist captcha_crnn_last.pth del /q captcha_crnn_last.pth
) else (
    echo [RESUME] Giu nguyen captcha_crnn_last.pth de tiep tuc training.
    echo [RESUME] Giu nguyen captcha_crnn_last.pth de tiep tuc training. >> "%LOGFILE%"
    if not exist captcha_crnn_last.pth (
        echo [ERROR] --resume yeu cau captcha_crnn_last.pth ton tai. Khong tim thay.
        goto error
    )
)
echo.

echo ============================================================
echo BUOC 1/3: Train CRNN+CTC (%IMG_COUNT% real, augment ON)
echo Args: %TRAIN_ARGS%
echo ============================================================
%PY% train_crnn.py %TRAIN_ARGS% --metrics-csv "%RUN_DIR%\metrics.csv" 2>&1 | %PY% -c "import sys; f=open(r'%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
if errorlevel 1 goto error
echo [OK] Buoc 1 hoan tat. Best model: captcha_crnn_model.pth (+ .onnx); Last: captcha_crnn_last.pth; Metrics: %RUN_DIR%\metrics.csv

echo.
echo ============================================================
echo BUOC 2/3: Eval tren %IMG_COUNT% anh real
echo ============================================================
%PY% eval_crnn.py --json-out "%RUN_DIR%\eval_summary.json" 2>&1 | %PY% -c "import sys; f=open(r'%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
if errorlevel 1 goto error
echo [OK] Buoc 2 hoan tat. Eval JSON: %RUN_DIR%\eval_summary.json

echo.
echo ============================================================
echo BUOC 3/3: Archive artifact vao %RUN_DIR%
echo ============================================================
if exist captcha_crnn_model.pth copy /y captcha_crnn_model.pth "%RUN_DIR%\" >nul
if exist captcha_crnn_last.pth  copy /y captcha_crnn_last.pth  "%RUN_DIR%\" >nul
if exist captcha_crnn_model.onnx copy /y captcha_crnn_model.onnx "%RUN_DIR%\" >nul
echo [OK] Da copy artifact vao %RUN_DIR%\

REM === Mirror log root tu run dir (de test_replay_log... va tools doc len) ===
copy /y "%LOGFILE%" train_log.txt >nul 2>&1

echo.
echo ============================================================
echo [DONE] Workflow hoan tat!
echo ============================================================
echo Finished: %date% %time%
echo Finished: %date% %time% >> "%LOGFILE%"
echo.
echo Run archive:  %RUN_DIR%\
echo Best model:   captcha_crnn_model.pth  (root + run dir)
echo Last (resume): captcha_crnn_last.pth  (root + run dir)
echo ONNX:         captcha_crnn_model.onnx (root + run dir)
echo Metrics CSV:  %RUN_DIR%\metrics.csv
echo Eval JSON:    %RUN_DIR%\eval_summary.json
echo Log:          %LOGFILE%
echo.
echo Resume tiep theo: run_all.bat --resume --epochs 50
echo.
echo Neu accuracy ^< 90%%, thu them tu tu:
echo   1. %PY% generate_synthetic_crnn.py --count 100000
echo   2. %PY% train_crnn.py --use-synthetic
echo   3. %PY% self_train.py
pause
goto :eof

:error
echo.
echo ============================================================
echo [ERROR] Co loi xay ra! Xem log: %LOGFILE%
echo ============================================================
echo Error at: %date% %time% >> "%LOGFILE%"
copy /y "%LOGFILE%" train_log.txt >nul 2>&1
pause
exit /b 1
