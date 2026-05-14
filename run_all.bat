@echo off
REM ============================================================================
REM run_all.bat - Chay toan bo workflow training tu dau den cuoi
REM ============================================================================
REM - Xoa model cu + train lai tu dau moi lan chay
REM - Tu dong detect so anh trong data/ (khong hardcode)
REM - Dung ngay khi co loi
REM - Log toan bo output ra file train_log.txt
REM ============================================================================

setlocal enabledelayedexpansion

REM === Log file ===
set LOGFILE=train_log.txt
echo ============================================================ > %LOGFILE%
echo CAPTCHA SOLVER - FULL TRAINING WORKFLOW >> %LOGFILE%
echo Started: %date% %time% >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo ============================================================
echo CAPTCHA SOLVER - FULL TRAINING WORKFLOW
echo ============================================================
echo Log file: %LOGFILE%
echo.

REM === Dem so anh trong data/ ===
set /a IMG_COUNT=0
for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
echo [INFO] Tim thay %IMG_COUNT% anh trong data/
echo [INFO] Tim thay %IMG_COUNT% anh trong data/ >> %LOGFILE%

if %IMG_COUNT% LSS 100 (
    echo [ERROR] Can it nhat 100 anh de train. Hien chi co %IMG_COUNT%.
    echo [ERROR] Can it nhat 100 anh de train. >> %LOGFILE%
    goto error
)

REM === Xoa model cu de train lai tu dau ===
echo.
echo [CLEAN] Xoa model cu...
echo [CLEAN] Xoa model cu... >> %LOGFILE%
if exist captcha_unet_model.pth del /q captcha_unet_model.pth
if exist captcha_trocr_model rmdir /s /q captcha_trocr_model
if exist data\real_backgrounds rmdir /s /q data\real_backgrounds
if exist data\unet_pairs rmdir /s /q data\unet_pairs
if exist data\synthetic rmdir /s /q data\synthetic
echo [CLEAN] Done.
echo.

REM === Import data moi tu dataset/ (neu co) ===
if exist dataset (
    echo ============================================================
    echo BUOC 0: Import data moi tu dataset/
    echo ============================================================
    python import_new_data.py >> %LOGFILE% 2>&1
    if errorlevel 1 goto error
    REM Re-count
    set /a IMG_COUNT=0
    for %%f in (data\map_*.png) do set /a IMG_COUNT+=1
    echo [INFO] Sau import: %IMG_COUNT% anh
    echo [INFO] Sau import: %IMG_COUNT% anh >> %LOGFILE%
)

echo.
echo ============================================================
echo BUOC 1/6: Extract real backgrounds tu %IMG_COUNT% anh
echo ============================================================
python extract_real_backgrounds.py >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 1 hoan tat.

echo.
echo ============================================================
echo BUOC 2/6: Generate U-Net training data
echo ============================================================
python generate_unet_data.py >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 2 hoan tat.

echo.
echo ============================================================
echo BUOC 3/6: Generate TrOCR synthetic data (co label)
echo ============================================================
python generate_trocr_synthetic.py >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 3 hoan tat.

echo.
echo ============================================================
echo BUOC 4/6: Train U-Net Denoiser
echo ============================================================
python train_unet.py >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 4 hoan tat. Model: captcha_unet_model.pth

echo.
echo ============================================================
echo BUOC 5/6: Train TrOCR (Combine synthetic + %IMG_COUNT% real)
echo ============================================================
python train.py --use-real-data --combine --augment >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 5 hoan tat. Model: captcha_trocr_model/

echo.
echo ============================================================
echo BUOC 6/6: Evaluate model tren %IMG_COUNT% anh real
echo ============================================================
python eval_model.py >> %LOGFILE% 2>&1
if errorlevel 1 goto error
echo [OK] Buoc 6 hoan tat.
echo.
echo --- KET QUA EVALUATE ---
python eval_model.py

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
echo.
echo Xem ket qua: python eval_model.py
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
