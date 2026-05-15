@echo off
REM ============================================================================
REM run_smoke.bat — Smoke test nhanh cho may CPU (verify code chay duoc)
REM ============================================================================
REM Muc dich: Chay nhanh ~30-60 phut tren CPU de verify pipeline khong crash.
REM           KHONG ky vong dat 90%, chi xem code chay het workflow.
REM
REM Khac voi run_all.bat:
REM   - Epochs giam tu 50 -> 5
REM   - Batch giam tu 64 -> 16 (it RAM hon)
REM   - Augment OFF (giam noise dau train)
REM
REM Output: smoke_log.txt (tach rieng, khong de len train_log.txt)
REM ============================================================================

setlocal enabledelayedexpansion

set PY=python
if exist "venv\Scripts\python.exe" (
    set PY=venv\Scripts\python.exe
    echo [INFO] Su dung venv: %PY%
) else (
    echo [WARN] Khong co venv, dung: %PY%
)

set LOGFILE=smoke_log.txt
echo ============================================================ > %LOGFILE%
echo CRNN+CTC SMOKE TEST (5 epochs, batch 16) >> %LOGFILE%
echo Started: %date% %time% >> %LOGFILE%
echo Python:  %PY% >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo ============================================================
echo CRNN+CTC SMOKE TEST (~30-60 phut tren CPU)
echo Muc dich: VERIFY code chay duoc, KHONG do accuracy
echo ============================================================
echo Log: %LOGFILE%
echo.

REM === Dem so anh real ===
set /a IMG_COUNT=0
for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
echo [INFO] Co %IMG_COUNT% anh real trong data/
echo [INFO] Co %IMG_COUNT% anh real trong data/ >> %LOGFILE%

if %IMG_COUNT% LSS 50 (
    echo [ERROR] Can it nhat 50 anh real. Hien chi co %IMG_COUNT%.
    goto error
)

REM === Xoa model smoke cu ===
if exist captcha_crnn_model.pth del /q captcha_crnn_model.pth
if exist captcha_crnn_model.onnx del /q captcha_crnn_model.onnx
if exist captcha_crnn_last.pth del /q captcha_crnn_last.pth

echo ============================================================
echo BUOC 1/2: Train smoke (5 epochs, batch 16, no augment)
echo ============================================================
powershell -Command "%PY% train_crnn.py --epochs 5 --batch-size 16 --no-augment 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Train smoke hoan tat.

echo.
echo ============================================================
echo BUOC 2/2: Eval smoke (truoc tien gioi xem so so the nao)
echo ============================================================
powershell -Command "%PY% eval_crnn.py --batch-size 16 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error

echo.
echo ============================================================
echo [DONE] Smoke test hoan tat!
echo ============================================================
echo Finished: %date% %time%
echo Finished: %date% %time% >> %LOGFILE%
echo.
echo Neu thay loss giam + co prediction (du sai), code OK.
echo Push code len git va chay run_all.bat tren may GPU.
pause
goto :eof

:error
echo.
echo ============================================================
echo [ERROR] Smoke test loi! Xem %LOGFILE%
echo ============================================================
echo Error at: %date% %time% >> %LOGFILE%
pause
exit /b 1
