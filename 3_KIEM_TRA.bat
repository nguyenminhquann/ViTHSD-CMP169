@echo off
REM BUOC 3 - Kiem tra thu vien, du lieu va 2 mo hinh da san sang chua
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Chua cai dat. Hay chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python kiem_tra.py
pause
