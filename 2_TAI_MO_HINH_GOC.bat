@echo off
REM ==================================================================
REM  BUOC 2 - TAI TU HUGGING FACE (CAN INTERNET, chi 1 lan):
REM    du lieu ViTHSD (can token Hugging Face) vao data\
REM    2 mo hinh goc ViSoBERT + PhoBERT-v2 (~1 GB) vao models\pretrained\
REM  Chay SAU 1_CAI_DAT.bat.
REM ==================================================================
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Chua cai dat. Hay chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python tai_mo_hinh_goc.py
if errorlevel 1 (
    echo ===== CO LOI - kiem tra Internet va token Hugging Face roi chay lai, hoac chup man hinh gui thay =====
)
pause
