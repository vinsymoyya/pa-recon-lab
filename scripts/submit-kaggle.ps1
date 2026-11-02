param([Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Owner)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:KAGGLE_CONFIG_DIR = Join-Path $env:TEMP 'pai-kaggle-config'
$metadataPath = Join-Path $projectRoot 'kaggle_kernel\kernel-metadata.json'
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
$metadata.id = "$Owner/pa-recon-robust-synthetic-pilot"
$json = $metadata | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($metadataPath, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
& (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $projectRoot 'scripts\build_kaggle_notebook.py')
if ($LASTEXITCODE -ne 0) { throw 'Notebook generation failed.' }
& (Join-Path $projectRoot '.venv\Scripts\kaggle.exe') kernels push -p (Join-Path $projectRoot 'kaggle_kernel') --timeout 3600
if ($LASTEXITCODE -ne 0) { throw 'Kaggle submission failed. Check authentication and accelerator availability.' }
