@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%..\embodied_ai_sim_platform.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%..\embodied_ai_sim_platform.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found.
    echo Looked for:
    echo   %ROOT%.venv\Scripts\python.exe
    echo   %ROOT%..\embodied_ai_sim_platform.venv\Scripts\python.exe
    echo See README.md section "Environment" to create it.
    pause
    exit /b 1
)

pushd "%ROOT%"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo  Embodied AI Sim Platform - live MuJoCo demo (real physics)
echo  A viewer window will open; close it to stop the demo.
echo ============================================================
echo.

"%PY%" "%ROOT%scripts\viewer_demo.py" %*
set "RC=%ERRORLEVEL%"

echo.
echo Done. exit code = %RC%
popd
pause
exit /b %RC%
