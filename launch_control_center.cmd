@echo off
cd /d "%~dp0"
python -c "import PyQt6" >nul 2>nul
if errorlevel 1 (
    echo PyQt6 is required. Run: python -m pip install -r requirements-gui.txt
    pause
    exit /b 1
)

for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
set "PYTHONW_EXE=%PYTHON_EXE:python.exe=pythonw.exe%"
if exist "%PYTHONW_EXE%" (
    start "" "%PYTHONW_EXE%" -X utf8 src\control_center.py
) else (
    python -X utf8 src\control_center.py
)
