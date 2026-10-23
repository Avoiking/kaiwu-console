@echo off
rem ============================================================
rem  kaiwu console launcher
rem  Auto-detects Python so it works on any machine.
rem  (Keep this file ASCII-only to avoid codepage issues.)
rem ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
title kaiwu console

rem --- run from this script's own folder (not the caller's cwd) ---
cd /d "%~dp0"

rem --- pick a working Python: env var > py launcher > python on PATH ---
set "PY="

if defined KAIWU_PYTHON (
    if exist "%KAIWU_PYTHON%" set "PY=%KAIWU_PYTHON%"
)

if not defined PY (
    py -3 --version >nul 2>&1 && set "PY=py -3"
)

if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo.
    echo   [X] Python not found.
    echo.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add Python to PATH" during setup.
    echo.
    echo   Already installed but not on PATH? Set KAIWU_PYTHON to the full
    echo   path of python.exe, e.g.:
    echo     set KAIWU_PYTHON=C:\Python312\python.exe
    echo.
    pause
    exit /b 1
)

rem --- make sure requests is available ---
%PY% -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Installing dependency 'requests' ...
    echo.
    %PY% -m pip install requests
    if errorlevel 1 (
        echo.
        echo   [X] Failed to install requests. Try running:
        echo         %PY% -m pip install requests
        echo.
        pause
        exit /b 1
    )
)

rem --- launch ---
echo.
echo   Starting kaiwu console ...
echo   Web UI will be at http://127.0.0.1:8080
echo   Close this window to stop.
echo.
%PY% console.py

echo.
echo Console exited. Press any key to close...
pause >nul
