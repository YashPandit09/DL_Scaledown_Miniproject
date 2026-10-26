@echo off
title Push SpaceGen AI to GitHub
echo ========================================================
echo   Pushing to https://github.com/YashPandit09/DL_Scaledown_Miniproject
echo ========================================================
echo.

set GIT_EXE=C:\Users\Thrill3r\AppData\Local\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd\git.exe

if not exist "%GIT_EXE%" (
    set GIT_EXE=git
)

"%GIT_EXE%" status
echo.
echo Pushing branch 'main' to GitHub...
echo (If prompted, enter your GitHub Username and Personal Access Token)
echo.

"%GIT_EXE%" push -u origin main

echo.
if %ERRORLEVEL% equ 0 (
    echo ========================================================
    echo   SUCCESS! Pushed successfully to GitHub!
    echo ========================================================
) else (
    echo [ERROR] Push failed. If authentication failed, use a Personal Access Token (PAT):
    echo https://github.com/settings/tokens
)

echo.
pause
