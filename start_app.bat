@echo off
rem Mosaic GUI launcher: double-click this file.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
echo Using Python: %PY%

"%PY%" -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo gradio is not installed yet. Installing it now (first time only^)...
    "%PY%" -m pip install "gradio>=5,<6"
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not install gradio. Check the internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting the app. The browser opens automatically.
echo To quit: close this window (or press Ctrl+C here).
"%PY%" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The app stopped with an error. See the messages above.
)
pause
