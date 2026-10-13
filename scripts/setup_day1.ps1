<#
.SYNOPSIS
    Day 1 setup: checks prerequisites, creates the venv, installs PyTorch,
    runs the CUDA verification and makes the first Git commit.

.DESCRIPTION
    Run it in two passes, from anywhere:

    Pass 1 (no arguments) - checks your machine, creates the venv, upgrades pip,
    then stops and tells you to look up the PyTorch install URL:
        powershell -ExecutionPolicy Bypass -File scripts\setup_day1.ps1

    Pass 2 - same script plus the index URL taken from the command generated at
    https://pytorch.org/get-started/locally/ (the part after --index-url):
        powershell -ExecutionPolicy Bypass -File scripts\setup_day1.ps1 `
            -TorchIndexUrl "https://download.pytorch.org/whl/cuXXX"

    The script is safe to re-run: finished steps are skipped.

    Why the URL is not hard-coded: PyTorch changes its supported CUDA builds
    regularly. A stale URL either fails or installs a CPU-only build.
#>
param(
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Tool([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' was not found. $Hint"
    }
}

# ---------------------------------------------------------------- Step 1
Write-Step "1/6 Checking prerequisites"
Test-Tool "nvidia-smi" "Install/update the NVIDIA driver from nvidia.com (Download Drivers)."
Test-Tool "python"     "Install Python 3.10 or 3.11 from python.org and tick 'Add to PATH'."
Test-Tool "git"        "Install Git for Windows from git-scm.com."

$gpu = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Write-Host "GPU, driver, VRAM : $gpu"

$pyVersion = python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "System Python     : $pyVersion"
$parts = $pyVersion.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    throw "Python 3.10 or newer is required (found $pyVersion)."
}

# ---------------------------------------------------------------- Step 2
Write-Step "2/6 Creating the virtual environment"
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "venv already exists - skipping."
} else {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) { throw "Creating the venv failed." }
    Write-Host "venv created."
}
# We call the venv's python.exe directly. That is equivalent to activating the
# venv and avoids PowerShell execution-policy problems with Activate.ps1.

# ---------------------------------------------------------------- Step 3
Write-Step "3/6 Upgrading pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# ---------------------------------------------------------------- Step 4
Write-Step "4/6 Installing PyTorch (CUDA build)"
if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) {
    Write-Host "No -TorchIndexUrl given, so stopping after the venv is ready." -ForegroundColor Yellow
    Write-Host "Next:"
    Write-Host "  1. Open https://pytorch.org/get-started/locally/"
    Write-Host "  2. Choose: Stable, Windows, Pip, Python, newest CUDA option"
    Write-Host "  3. Copy the URL after --index-url in the generated command"
    Write-Host "  4. Re-run this script with -TorchIndexUrl <that url>"
    exit 0
}
& $VenvPython -m pip install torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed. Check the URL." }

# ---------------------------------------------------------------- Step 5
Write-Step "5/6 Verifying the GPU"
& $VenvPython (Join-Path $ProjectRoot "scripts\verify_cuda.py")
$verifyOk = ($LASTEXITCODE -eq 0)

# ---------------------------------------------------------------- Step 6
Write-Step "6/6 Freezing dependencies and committing to Git"
& $VenvPython -m pip freeze | Set-Content -Path (Join-Path $ProjectRoot "requirements.txt") -Encoding ascii

if (-not (Test-Path (Join-Path $ProjectRoot ".git"))) {
    git init
    git branch -M main
}

if (-not $verifyOk) {
    Write-Host "GPU verification did not fully pass, so no commit was made." -ForegroundColor Yellow
    Write-Host "Fix the issue shown above, then re-run this script."
    exit 1
}

$gitName = git config user.name
$gitEmail = git config user.email
if ([string]::IsNullOrWhiteSpace($gitName) -or [string]::IsNullOrWhiteSpace($gitEmail)) {
    Write-Host "Git does not know who you are yet. Run these once, then re-run the script:" -ForegroundColor Yellow
    Write-Host '  git config --global user.name  "Your Name"'
    Write-Host '  git config --global user.email "you@example.com"'
    exit 1
}

git add .
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "Nothing new to commit."
} else {
    git commit -m "Day 1: project structure, venv setup and CUDA verification"
}

Write-Host ""
Write-Host "Day 1 setup complete." -ForegroundColor Green
