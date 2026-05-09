@echo off
set PYTHON=C:\Users\zhangyicong2\.workbuddy\binaries\python\versions\3.13.12\python.exe
set WORKDIR=C:\Users\zhangyicong2\WorkBuddy\20260506224350

cd /d "%WORKDIR%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Cannot find directory %WORKDIR%
    pause
    exit /b 1
)

echo ========================================
echo   PPT内嵌图片提取器
echo ========================================
echo.
"%PYTHON%" simple_extractor.py
echo.
echo ========================================
pause
