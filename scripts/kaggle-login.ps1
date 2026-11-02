$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'PA Recon Lab | Kaggle login'
$projectRoot = Split-Path -Parent $PSScriptRoot
$kaggleCli = Join-Path $projectRoot '.venv\Scripts\kaggle.exe'
if (-not (Test-Path -LiteralPath $kaggleCli)) {
    throw 'Install Kaggle in the project environment: .\.venv\Scripts\python.exe -m pip install kaggle'
}
$env:KAGGLE_CONFIG_DIR = Join-Path $env:TEMP 'pai-kaggle-config'
Write-Host 'Complete the Kaggle browser login. Credentials stay outside the repository.'
& $kaggleCli auth login
if ($LASTEXITCODE -ne 0) { throw 'Kaggle login did not complete.' }
Write-Host 'Kaggle login completed. You can return to the research task.'
