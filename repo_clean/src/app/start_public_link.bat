@echo off
rem MT2KD public demo link: starts the app (if not already running) and opens a free Cloudflare quick tunnel.
rem Needs cloudflared.exe in this folder (or installed). Keep this window open while people use the link.
cd /d "%~dp0"
set CF=cloudflared.exe
if not exist cloudflared.exe (
  where cloudflared >nul 2>nul || (
    echo.
    echo  cloudflared.exe was not found.
    echo  Download cloudflared-windows-amd64.exe from https://github.com/cloudflare/cloudflared/releases/latest
    echo  rename it to cloudflared.exe and put it in this folder, then run this file again.
    echo.
    pause
    exit /b 1
  )
  set CF=cloudflared
)
curl -s http://127.0.0.1:8000/api/health >nul 2>nul
if errorlevel 1 (
  echo Starting the MT2KD app in a second window ...
  start "MT2KD app" cmd /k run_windows.bat
  echo Waiting 40 s for the model to load ...
  timeout /t 40 /nobreak >nul
)
echo.
echo ================================================================================
echo  Your PUBLIC link appears below as  https://....trycloudflare.com
echo  Share it, it works on phones (microphone too). Close this window to stop sharing.
echo  The link is new every time you run this file.
echo ================================================================================
echo.
echo  Starting the tunnel ... the link appears in 5-20 s (only the link and errors are shown here).
echo.
%CF% tunnel --url http://127.0.0.1:8000 2>&1 | findstr /i /c:"trycloudflare.com " /c:"ERR"
pause
