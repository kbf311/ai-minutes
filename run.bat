@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================
echo   AI Minutes (話者分離・議事録作成システム)
echo ========================================================
python main.py %*
pause
