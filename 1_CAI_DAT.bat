@echo off
REM ==================================================================
REM  BUOC 1 - CAI DAT MOI TRUONG (chi lam 1 lan, CAN INTERNET)
REM  Nhap dup file nay. Mat khoang 10-20 phut tuy toc do mang.
REM  Tu nhan card NVIDIA: co GPU -> cai PyTorch GPU, khong co -> ban CPU.
REM ==================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] Kiem tra Python 3.12 ...
py -3.12 --version
if errorlevel 1 (
    echo.
    echo LOI: Chua co Python 3.12. Tai tai https://www.python.org/downloads/
    echo      Khi cai nho tick o "Add python.exe to PATH", roi chay lai file nay.
    pause
    exit /b 1
)

echo [2/4] Tao moi truong ao .venv ...
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/4] Cai PyTorch ...
set TORCH_IDX=cpu
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set TORCH_IDX=cu126
    nvidia-smi | findstr /C:"CUDA Version: 13" >nul
    if not errorlevel 1 set TORCH_IDX=cu130
)
echo     -> ban PyTorch: %TORCH_IDX%   (cpu = khong co card NVIDIA)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/%TORCH_IDX%
if errorlevel 1 goto :error

echo [4/4] Cai cac thu vien con lai ...
pip install -r requirements_local.txt
if errorlevel 1 goto :error

echo.
echo ===== CAI DAT XONG. Tiep theo: nhap dup 2_TAI_MO_HINH_GOC.bat. =====
pause
exit /b 0

:error
echo.
echo ===== CO LOI - chup man hinh cua so nay gui thay =====
pause
exit /b 1
