@echo off
echo --- LAIXR Handtracking Setup ---

where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    echo    Download from: https://nodejs.org/
    pause
    exit /b 1
)

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python is not installed. Please install Python first.
    echo    Download from: https://www.python.org/
    pause
    exit /b 1
)

echo ✅ Prerequisites found

echo 📦 Installing frontend dependencies...
npm install
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

echo 🐍 Creating Python virtual environment...
python -m venv venv

echo 📦 Installing backend dependencies...
call venv\Scripts\activate
pip install -r backend\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Setup complete!
echo.
echo To start the application, run start.bat
echo.
pause 