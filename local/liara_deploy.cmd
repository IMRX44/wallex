@echo off
rem Deploy the current code to the Liara app "pishro" WITHOUT changing its mode (paper stays paper,
rem live stays live). Reads LIARA_API_TOKEN / LIARA_TEAM_ID from .env.
rem Run it yourself:   P:\local\liara_deploy.cmd
setlocal
cd /d "%~dp0.."
set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" ( echo Git Bash not found at %BASH% & exit /b 1 )
"%BASH%" -c "set -a; . <(grep -E '^LIARA_(API_TOKEN|TEAM_ID)=' .env | tr -d '\r'); set +a; APP=pishro bash scripts/deploy_liara.sh < /dev/null 2>&1 | grep -E 'deployed|Error|successfully|restarting'; git checkout -q -- liara.json 2>/dev/null"
echo.
echo Deploy sent. The app restarts with the new code in 1-3 minutes: https://pishro.liara.run
endlocal
