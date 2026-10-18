@echo off
setlocal
cd /d "%~dp0"
title GreenGuardian AI

echo ============================================
echo GreenGuardian AI - Final CNN Ready Project
echo ============================================

if exist "venv\Scripts\python.exe" goto HAVEVENV

echo Creating Python environment...
py -3.12 -m venv venv 2>nul
if exist "venv\Scripts\python.exe" goto HAVEVENV
py -3.11 -m venv venv 2>nul
if exist "venv\Scripts\python.exe" goto HAVEVENV
py -3.10 -m venv venv 2>nul
if exist "venv\Scripts\python.exe" goto HAVEVENV

echo ERROR: Please install Python 3.10, 3.11 or 3.12.
pause
exit /b 1

:HAVEVENV
call "venv\Scripts\activate.bat"
python -c "import streamlit,pandas,PIL,reportlab,numpy,tensorflow" >nul 2>&1
if errorlevel 1 (
    echo Installing application packages. This may take several minutes...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 goto FAIL
)

echo Starting GreenGuardian AI...
python -m streamlit run app.py
pause
exit /b 0

:FAIL
echo Package installation failed. Send a screenshot of this window.
pause
exit /b 1
