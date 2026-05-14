@echo off
REM ============================================================================
REM run_all.bat - Chay toan bo workflow training tu dau den cuoi
REM ============================================================================
REM Yeu cau:
REM   - Da chay 'venv\Scripts\activate' va cai requirements.txt
REM   - Da co data/metadata.csv va 500 anh data/map_*.png
REM ============================================================================

echo ============================================================
echo BUOC 1/5: Extract real backgrounds tu 500 anh
echo ============================================================
python extract_real_backgrounds.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo BUOC 2/5: Generate U-Net training data (12K pairs)
echo ============================================================
python generate_unet_data.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo BUOC 3/5: Train U-Net Denoiser (~10-15 phut)
echo ============================================================
python train_unet.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo BUOC 4/5: Train TrOCR voi U-Net preprocessing (~30-60 phut)
echo ============================================================
python train.py --use-real-data --augment
if errorlevel 1 goto error

echo.
echo ============================================================
echo BUOC 5/5: Evaluate model tren 500 anh real
echo ============================================================
python evaluate.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo [DONE] Workflow hoan tat!
echo ============================================================
echo Xem ket qua evaluation o tren.
echo Model TrOCR luu o: ./captcha_trocr_model/
echo Model U-Net luu o: ./captcha_unet_model.pth
goto :eof

:error
echo.
echo [ERROR] Co loi xay ra. Xem log de debug.
exit /b 1
