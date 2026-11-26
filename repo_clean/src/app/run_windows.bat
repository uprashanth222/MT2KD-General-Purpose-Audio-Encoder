@echo off
rem MT2KD web app: first run installs everything into .venv (a few minutes), later runs start in seconds.
cd /d "%~dp0"

if not exist deploy\model.pt (
  echo.
  echo  deploy\model.pt was not found.
  echo  Download the folder MT2KD/deploy/mt2kd_S20_main from Google Drive and put its files into the "deploy" folder here.
  echo.
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo Creating the Python environment in .venv ...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist .venv\Scripts\python.exe (
    echo Could not create the environment. Is Python installed and on PATH?
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r backend\requirements.txt
  if errorlevel 1 (
    echo Installing the packages failed. See the message above.
    pause
    exit /b 1
  )
)

echo Starting MT2KD on http://localhost:8000  (close this window to stop it)
start "" http://localhost:8000
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause
