@echo off
setlocal
cd /d "%~dp0"

python -m PyInstaller --noconfirm --clean NovelSpeakerControlCenter.spec
if errorlevel 1 exit /b %errorlevel%

echo.
echo Built: %~dp0dist\NovelSpeakerControlCenter.exe
endlocal
