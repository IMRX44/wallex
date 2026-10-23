# Start Pishro with REAL orders on the Wallex challenge account (Windows PowerShell).
# Run it yourself:   powershell -ExecutionPolicy Bypass -File local\live.ps1
# Stop it:           Ctrl+C  (open orders are cancelled on a clean shutdown;
#                    after a crash they are cancelled at the next start)
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONUTF8   = "1"
$env:PISHRO_MODE  = "live"
$env:LIVE_CONFIRM = "I_UNDERSTAND"
$env:LOG_FILE     = "live.log"

Write-Host "Pishro LIVE - real orders. Dashboard: http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Yellow
& $py -m pishro serve --host 127.0.0.1 --port 8000
