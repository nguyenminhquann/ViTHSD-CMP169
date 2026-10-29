@echo off
REM BUOC 5 - Mo demo web (chay OFFLINE) tai http://localhost:8501
REM Dung demo: bam Ctrl + C trong cua so nay.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Chua cai dat. Hay chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo Dang mo demo... neu trinh duyet bao loi, doi vai giay roi bam F5.
start "" http://localhost:8501
streamlit run app.py
pause
