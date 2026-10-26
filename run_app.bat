@echo off
title SpaceGen AI - Deep Learning Layout Generator
echo ========================================================
echo   Starting SpaceGen AI Web Demo (Streamlit)
echo ========================================================
echo.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] .venv not found, trying system Python...
)

echo Opening web interface in browser...
streamlit run app/streamlit_app.py
pause
