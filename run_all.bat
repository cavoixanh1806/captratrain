@echo off
REM ============================================================================
REM run_all.bat — CRNN+CTC pipeline (toi gian, full epochs)
REM ============================================================================
REM Auto-detect venv\Scripts\python.exe, fallback `python`.
REM Output: vua hien CMD vua ghi train_log.txt (PowerShell Tee-Object).
REM
REM Workflow:
REM   0. Import anh moi tu dataset/ neu co
REM   1. Train CRNN+CTC tren 754 real (val 15%, augment ON)
REM   2. Eval tren 754 real
REM
REM TAM THOI KHONG bat:
REM   - Synthetic (chay rieng: python generate_synthetic_crnn.py)
REM   - Self-train (chay rieng: python self_train.py)
REM
REM Thoi gian: ~30-45 phut RTX 3060, ~10-15h CPU
REM ============================================================================

setlocal enabledelayedexpansion

REM === Auto-detect Python (uu tien venv local) ===
set PY=python
if exist "venv\Scripts\python.exe" (
    set PY=venv\Scripts\python.exe
    echo [INFO] Su dung venv: %PY%
) else (
    echo [WARN] Khong tim thay venv\Scripts\python.exe, fallback ve: %PY%
)

set LOGFILE=train_log.txt
echo ============================================================ > %LOGFILE%
echo CRNN+CTC CAPTCHA - TRAINING WORKFLOW (minimal) >> %LOGFILE%
echo Started: %date% %time% >> %LOGFILE%
echo Python:  %PY% >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo ============================================================
echo CRNN+CTC CAPTCHA - TRAINING WORKFLOW (minimal)
echo ============================================================
echo Log file: %LOGFILE%
echo.

REM === Buoc 0: Import data moi tu dataset/ ===
if exist dataset (
    echo BUOC 0/2: Import data moi tu dataset/
    %PY% import_new_data.py 2>&1 | %PY% -c "import sys; f=open('%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
    if errorlevel 1 goto error
    echo [OK] Import xong.
    echo.
)

REM === Dem so anh real ===
set /a IMG_COUNT=0
for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
echo [INFO] Co %IMG_COUNT% anh real trong data/
echo [INFO] Co %IMG_COUNT% anh real trong data/ >> %LOGFILE%

if %IMG_COUNT% LSS 50 (
    echo [ERROR] Can it nhat 50 anh real de train. Hien chi co %IMG_COUNT%.
    goto error
)

REM === Xoa model cu (clean train) ===
echo [CLEAN] Xoa model cu...
echo [CLEAN] Xoa model cu... >> %LOGFILE%
if exist captcha_crnn_model.pth del /q captcha_crnn_model.pth
if exist captcha_crnn_model.onnx del /q captcha_crnn_model.onnx
if exist captcha_crnn_last.pth del /q captcha_crnn_last.pth
echo.

echo ============================================================
echo BUOC 1/2: Train CRNN+CTC (200 epochs, %IMG_COUNT% real, augment ON)
echo ============================================================
%PY% train_crnn.py 2>&1 | %PY% -c "import sys; f=open('%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
if errorlevel 1 goto error
echo [OK] Buoc 1 hoan tat. Model: captcha_crnn_model.pth (+ .onnx)

echo.
echo ============================================================
echo BUOC 2/2: Eval tren %IMG_COUNT% anh real
echo ============================================================
%PY% eval_crnn.py 2>&1 | %PY% -c "import sys; f=open('%LOGFILE%', 'a', encoding='utf-8'); [(sys.stdout.write(l), f.write(l), f.flush()) for l in sys.stdin]"
if errorlevel 1 goto error
echo [OK] Buoc 2 hoan tat.

echo.
echo ============================================================
echo [DONE] Workflow hoan tat!
echo ============================================================
echo Finished: %date% %time%
echo Finished: %date% %time% >> %LOGFILE%
echo.
echo Model:    captcha_crnn_model.pth
echo ONNX:     captcha_crnn_model.onnx
echo Log:      %LOGFILE%
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
echo Error at: %date% %time% >> %LOGFILE%
pause
exit /b 1
