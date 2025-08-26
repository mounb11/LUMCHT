Write-Host "--- LAIXR Handtracking Setup ---" -ForegroundColor Green

# Check for Node.js
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js is not installed. Please install Node.js first." -ForegroundColor Red
    Write-Host "   Download from: https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
    exit 1
}

# Check for Python
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed. Please install Python first." -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "✅ Prerequisites found" -ForegroundColor Green

Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install frontend dependencies" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "🐍 Creating Python virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host "📦 Installing backend dependencies..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"
pip install -r backend\requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install backend dependencies" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run: .\start-backend.ps1 in one terminal and .\start-frontend.ps1 in another, or double-click start.bat" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to continue" 