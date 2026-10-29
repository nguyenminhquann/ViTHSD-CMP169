@echo off
REM BUOC 4 - Mo Jupyter (chay OFFLINE). Mo lan luot:
REM   01_EDA_Baseline.ipynb -> 02_Transformer.ipynb -> 03_Demo_Offline.ipynb
REM Trong moi notebook: Run - Run All Cells
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Chua cai dat. Hay chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
jupyter notebook
pause
