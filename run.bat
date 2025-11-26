@echo off
REM Quick start script for Bangkok Traffy Dashboard (Windows)

echo =========================================
echo Bangkok Traffy Dashboard - Quick Start
echo =========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
    echo.
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
if not exist "venv\installed.flag" (
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    type nul > venv\installed.flag
    echo Dependencies installed
    echo.
)

REM Display options
echo Select mode:
echo   1) Dashboard (interactive web app)
echo   2) Generate maps
echo   3) Generate network graph
echo   4) Generate all visualizations
echo.
set /p choice="Enter choice [1-4]: "

REM Set sample size
set /p sample="Enter sample fraction (0.01 to 1.0) [default: 0.1]: "
if "%sample%"=="" set sample=0.1

REM Set CSV file
set /p csvfile="Enter CSV filename [default: bangkok_traffy_30.csv]: "
if "%csvfile%"=="" set csvfile=bangkok_traffy_30.csv

echo.
echo Starting with:
echo   CSV: %csvfile%
echo   Sample: %sample%
echo.

REM Run based on choice
if "%choice%"=="1" (
    echo Starting dashboard on http://localhost:8050
    echo Press Ctrl+C to stop
    python app.py dashboard --csv %csvfile% --sample %sample%
) else if "%choice%"=="2" (
    echo Generating maps...
    python app.py maps --csv %csvfile% --sample %sample%
) else if "%choice%"=="3" (
    echo Generating network graph...
    python app.py network --csv %csvfile% --sample %sample%
) else if "%choice%"=="4" (
    echo Generating all visualizations...
    python app.py all --csv %csvfile% --sample %sample%
) else (
    echo Invalid choice. Exiting.
    call deactivate
    exit /b 1
)

echo.
echo Done! Deactivating virtual environment...
call deactivate
