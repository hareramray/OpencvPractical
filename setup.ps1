$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the environment. Install Python 3.13 first.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection and the error above.' }
& '.\.venv\Scripts\python.exe' -m ipykernel install --prefix .venv --name opencv-practical --display-name 'Python (OpenCV Practical)'
if ($LASTEXITCODE -ne 0) { throw 'Kernel registration failed.' }
Write-Host 'Ready. Run .\start-learning.ps1 to open JupyterLab.'
