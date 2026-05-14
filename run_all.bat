@echo off
REM ============================================================================
REM run_all.bat - Chay toan bo workflow training tu dau den cuoi
REM ============================================================================
REM - Tu dong import anh moi tu dataset/ neu co
REM - Xoa model cu + data generated, train lai tu dau
REM - Dung ngay khi co loi (errorlevel)
REM - Log VUA hien CMD VUA ghi train_log.txt (dung PowerShell Tee-Object)
REM ============================================================================

setlocal enabledelayedexpansion

set LOGFILE=train_log.txt
echo ============================================================ > %LOGFILE%
echo CAPTCHA SOLVER - FULL TRAINING WORKFLOW >> %LOGFILE%
echo Started: %date% %time% >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo ============================================================
echo CAPTCHA SOLVER - FULL TRAINING WORKFLOW
echo ============================================================
echo Log file: %LOGFILE%
echo (Output hien o ca CMD lan log file)
echo.

REM === Import data moi tu dataset/ ===
if exist dataset (
    echo BUOC 0: Import data moi tu dataset/
    powershell -Command "python import_new_data.py 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
    if errorlevel 1 goto error
    echo [OK] Import xong.
    echo.
)

REM === Dem so anh trong data/ ===
set /a IMG_COUNT=0
for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
echo [INFO] Co %IMG_COUNT% anh real trong data/
echo [INFO] Co %IMG_COUNT% anh real trong data/ >> %LOGFILE%

if %IMG_COUNT% LSS 100 (
    echo [ERROR] Can it nhat 100 anh de train. Hien chi co %IMG_COUNT%.
    goto error
)

REM === Xoa model + data cu ===
echo [CLEAN] Xoa model + data cu...
echo [CLEAN] Xoa model + data cu... >> %LOGFILE%
if exist captcha_unet_model.pth del /q captcha_unet_model.pth
if exist captcha_trocr_model rmdir /s /q captcha_trocr_model
if exist data\unet_pairs rmdir /s /q data\unet_pairs
if exist data\synthetic rmdir /s /q data\synthetic
echo.

echo ============================================================
echo BUOC 1/5: Generate U-Net training data (12K pairs)
echo ============================================================
powershell -Command "python generate_unet_data.py 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Buoc 1 hoan tat.

echo.
echo ============================================================
echo BUOC 2/5: Generate TrOCR synthetic data (2K labeled samples)
echo ============================================================
powershell -Command "python generate_trocr_synthetic.py 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Buoc 2 hoan tat.

echo.
echo ============================================================
echo BUOC 3/5: Train U-Net Denoiser (~10-15 phut)
echo ============================================================
powershell -Command "python train_unet.py 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Buoc 3 hoan tat. Model: captcha_unet_model.pth

echo.
echo ============================================================
echo BUOC 4/5: Train TrOCR Combine synthetic + %IMG_COUNT% real (~5-7 gio)
echo ============================================================
powershell -Command "python train.py --use-real-data --combine --augment 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Buoc 4 hoan tat. Model: captcha_trocr_model/

echo.
echo ============================================================
echo BUOC 5/5: Evaluate model tren %IMG_COUNT% anh real
echo ============================================================
powershell -Command "python eval_model.py 2>&1 | Tee-Object -FilePath %LOGFILE% -Append"
if errorlevel 1 goto error
echo [OK] Buoc 5 hoan tat.

echo.
echo ============================================================
echo [DONE] Workflow hoan tat!
echo ============================================================
echo Finished: %date% %time%
echo Finished: %date% %time% >> %LOGFILE%
echo.
echo Model TrOCR: ./captcha_trocr_model/
echo Model U-Net: ./captcha_unet_model.pth
echo Log:         %LOGFILE%
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
