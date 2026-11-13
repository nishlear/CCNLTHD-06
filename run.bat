@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-app.ps1" %*
set "app_exit_code=%errorlevel%"
if not "%app_exit_code%"=="0" (
    echo.
    echo Application did not start successfully. Review the message above.
    pause
)
exit /b %app_exit_code%
