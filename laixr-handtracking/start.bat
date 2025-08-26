@echo off
echo --- Starting LAIXR Handtracking Application ---

echo [1/2] Starting Python backend...
set "VENV_PY=venv\Scripts\python.exe"
for /f %%a in ('powershell -NoProfile -Command "$p=Start-Process -FilePath '%VENV_PY%' -ArgumentList '-m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir ./backend' -PassThru -WindowStyle Hidden; $p.Id"') do set BACKEND_PID=%%a
echo Backend server started (PID %BACKEND_PID%)

echo [2/2] Starting Next.js frontend...
set NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
set PORT=3000
echo Ensuring port 3000 is free...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host \"Stopping PID $_ on port 3000\"; Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }" >nul 2>&1
for /f %%a in ('powershell -NoProfile -Command "$p=Start-Process -FilePath 'cmd' -ArgumentList '/c npm run dev' -PassThru; $p.Id"') do set FRONTEND_PID=%%a
echo Frontend server started (PID %FRONTEND_PID%)

echo.
echo Servers are starting. Please wait a moment...
timeout /t 5 /nobreak >nul

echo.
echo --- LAIXR Application is Running ---
echo Backend available at:  http://localhost:8000
echo Frontend available at: http://localhost:3000
echo.
echo Press any key to stop both servers...
echo.

pause

echo Stopping servers...
if defined FRONTEND_PID taskkill /PID %FRONTEND_PID% /F >nul 2>&1
if defined BACKEND_PID taskkill /PID %BACKEND_PID% /F >nul 2>&1
echo Stopped.