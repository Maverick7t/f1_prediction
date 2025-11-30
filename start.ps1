# F1 Prediction System - Startup Script
# Run this script to start both backend and frontend

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "F1 Race Prediction System Startup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Backend setup
Write-Host "[1/5] Setting up Backend..." -ForegroundColor Yellow
$backendDir = Join-Path $scriptDir "backend"
Set-Location $backendDir

# Check if venv exists
$venvDir = "f1env"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Green
    python -m venv $venvDir
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\$venvDir\Scripts\Activate.ps1"

# Install requirements
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt --quiet

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from template..." -ForegroundColor Green
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "WARNING: Please update .env with your settings!" -ForegroundColor Yellow
    }
}

# Check if models exist
if (-not (Test-Path "models_spencer/metadata.joblib")) {
    Write-Host "WARNING: ML models not found! Please run training first." -ForegroundColor Red
    Write-Host "Open final.ipynb and run the training cells." -ForegroundColor Red
    Read-Host "Press Enter to continue anyway or Ctrl+C to exit"
}

Write-Host ""
Write-Host "[2/5] Starting Backend API Server..." -ForegroundColor Yellow
Write-Host "Backend will run on http://localhost:5000" -ForegroundColor Green

# Start backend in new window (using f1env)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; .\f1env\Scripts\Activate.ps1; python api.py"

# Wait for backend to start
Write-Host "Waiting for backend to start..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Frontend setup
Write-Host ""
Write-Host "[3/5] Setting up Frontend..." -ForegroundColor Yellow
$frontendDir = Join-Path $scriptDir "frontend"
Set-Location $frontendDir

# Check if .env exists for frontend
if (-not (Test-Path ".env")) {
    Write-Host "Creating frontend .env from template..." -ForegroundColor Green
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    }
}

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Green
    npm install
}

Write-Host ""
Write-Host "[4/5] Starting Frontend Dev Server..." -ForegroundColor Yellow
Write-Host "Frontend will run on http://localhost:5173" -ForegroundColor Green

# Start frontend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; npm run dev"

Write-Host ""
Write-Host "[5/5] Launch Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "System is now running!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API:  http://localhost:5000" -ForegroundColor White
Write-Host "Frontend App: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Environment:  Development" -ForegroundColor Gray
Write-Host "Config File:  backend\.env" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C in each terminal to stop the servers" -ForegroundColor Gray
Write-Host ""

# Open browser
Start-Sleep -Seconds 2
Write-Host "Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "Setup complete! Check the new terminal windows." -ForegroundColor Green
