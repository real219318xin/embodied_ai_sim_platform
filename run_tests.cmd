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
echo  Unit tests (scene, IK, Chinese parsing, scene graph, planner)
echo ============================================================
echo.

"%PY%" -m pytest "%ROOT%tests" -q
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo ALL TESTS PASSED
) else (
    echo TESTS FAILED - exit code %RC%
)
echo.
echo Full physical verification scripts ^(each runs a real simulation^):
echo   "%PY%" scripts\test_motion.py
echo   "%PY%" scripts\test_perception.py
echo   "%PY%" scripts\run_grasp.py
popd
pause
exit /b %RC%
