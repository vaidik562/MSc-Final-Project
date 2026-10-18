@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title GreenGuardian AI - Final CNN Training

echo ================================================================
echo GreenGuardian AI - FINAL STABLE CNN TRAINING
echo ================================================================
echo Final plants: Tomato, Apple, Rose, Grape, Corn
echo.
echo Dataset locations:
echo   PlantVillage : D:\Dataset\PlantVillage
echo   Rose         : D:\Dataset\Rose
echo   Apple        : D:\Dataset\Apple
echo   Grape        : D:\Dataset\Grape
echo.
echo AppleLeaf9 downloads from GitHub when missing.
echo Grape GLVD downloads directly from Zenodo when missing.
echo Rose is validated locally; if missing, the official page opens.
echo ================================================================
echo.

if not exist "D:\" (
    echo ERROR: D: drive is not available.
    pause
    exit /b 1
)

set "GG_TEMP=D:\GreenGuardian_Project\_training_temp"
set "PIP_CACHE_DIR=D:\GreenGuardian_Project\_pip_cache"
set "TEMP=%GG_TEMP%"
set "TMP=%GG_TEMP%"

if not exist "%GG_TEMP%" mkdir "%GG_TEMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "D:\Dataset\Apple" mkdir "D:\Dataset\Apple"
if not exist "D:\Dataset\Grape" mkdir "D:\Dataset\Grape"
if not exist "D:\Dataset\Rose" mkdir "D:\Dataset\Rose"

if not exist "D:\Dataset\PlantVillage" (
    echo ERROR: D:\Dataset\PlantVillage is missing.
    echo Put the extracted PlantVillage dataset there first.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [1/9] Creating Python environment on D:...
    py -3.12 -m venv venv 2>nul
    if not exist "venv\Scripts\python.exe" py -3.11 -m venv venv 2>nul
    if not exist "venv\Scripts\python.exe" py -3.10 -m venv venv 2>nul
)

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python 3.10, 3.11 or 3.12 is required.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

echo [2/9] Checking application packages...
python -c "import streamlit,pandas,PIL,reportlab,numpy" >nul 2>&1
if errorlevel 1 (
    python -m pip install -r requirements.txt
    if errorlevel 1 goto FAIL
)

echo [3/9] Checking CNN training packages...
python -c "import tensorflow,sklearn,matplotlib" >nul 2>&1
if errorlevel 1 (
    python -m pip install -r requirements-training.txt
    if errorlevel 1 goto FAIL
)

echo [4/9] Preparing Rose dataset...
python training\setup_rose_dataset.py
if errorlevel 1 goto FAIL

echo [5/9] Preparing AppleLeaf9...
python training\setup_apple_dataset.py
if errorlevel 1 goto FAIL

echo [6/9] Preparing Zenodo GLVD grape dataset...
python training\setup_grape_dataset.py
if errorlevel 1 goto FAIL

echo [7/9] Building deduplicated 70/15/15 datasets...
python training\auto_prepare_training.py
if errorlevel 1 goto FAIL

echo [8/9] Training five MobileNetV2 CNN models...
python training\train_models.py --plant all
if errorlevel 1 goto FAIL

echo [9/9] Final status...
python training\training_status.py

echo.
echo ================================================================
echo SUCCESS - FINAL CNN TRAINING COMPLETE
echo Models     : model\
echo Evaluation : training\results\
echo ================================================================
pause
exit /b 0

:FAIL
echo.
echo ================================================================
echo TRAINING STOPPED SAFELY
echo Read the exact message above.
echo No incomplete CNN model is accepted as final.
echo ================================================================
pause
exit /b 1
