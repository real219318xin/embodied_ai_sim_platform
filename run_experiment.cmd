@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%..\embodied_ai_sim_platform.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found: %PY%
    echo See README.md section "Environment" to create it.
    pause
    exit /b 1
)

pushd "%ROOT%"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo  Batch experiment: 5 seeds x 2 trials (headless physics)
echo  Outputs: results\experiment.csv, summary.csv, plots\*.png
echo ============================================================
echo.

"%PY%" "%ROOT%scripts\run_experiment.py" --seeds 1 2 3 4 5 --trials 2 %*
set "RC=%ERRORLEVEL%"

echo.
echo Done. exit code = %RC%
popd
pause
exit /b %RC%
