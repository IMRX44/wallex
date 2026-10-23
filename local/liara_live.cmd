@echo off
rem Switch the Liara app "pishro" from shadow (paper) to REAL orders on the Wallex challenge account.
rem Run it yourself:   P:\local\liara_live.cmd
rem Back to paper:     P:\local\liara_live.cmd paper
rem Reads LIARA_API_TOKEN / LIARA_TEAM_ID from .env (never typed on the command line).
setlocal
cd /d "%~dp0.."
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if "%%A"=="LIARA_API_TOKEN" set "LT=%%B"
  if "%%A"=="LIARA_TEAM_ID" set "LTEAM=%%B"
)
if "%LT%"=="" ( echo LIARA_API_TOKEN missing in .env & exit /b 1 )
if /i "%1"=="paper" (
  call liara.cmd env set PISHRO_MODE=paper LIVE_CONFIRM=off --app pishro --api-token %LT% --team-id %LTEAM% --force
) else (
  call liara.cmd env set PISHRO_MODE=live LIVE_CONFIRM=I_UNDERSTAND --app pishro --api-token %LT% --team-id %LTEAM% --force
)
echo.
echo Done. The app restarts in about a minute: https://pishro.liara.run
endlocal
