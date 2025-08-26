# Script to start the Next.js frontend
Write-Host "--- Starting Next.js Frontend Server ---" -ForegroundColor Green
Write-Host "This terminal will show the frontend logs."
Write-Host "URL: http://localhost:3000"
Write-Host "----------------------------------------"

# Add Node.js to the path for this session to ensure 'npm' is found
$env:PATH = "C:\Program Files\nodejs;" + $env:PATH

# Set backend URL for the frontend
if (-not $env:NEXT_PUBLIC_BACKEND_URL) {
  $env:NEXT_PUBLIC_BACKEND_URL = "http://localhost:8000"
}

# Force Next.js to use port 3000 and make sure it is free
$env:PORT = "3000"
try {
  Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Write-Host "Stopping PID $_ on port 3000"; Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
} catch {}

# Start the Next.js development server
npm run dev 