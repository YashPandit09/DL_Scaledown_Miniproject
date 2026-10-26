@echo off
title SpaceGen AI - Quick CLI Demo
echo ========================================================
echo   Running SpaceGen AI Quick Demo
echo ========================================================
echo.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python quick_demo.py
echo.
pause
