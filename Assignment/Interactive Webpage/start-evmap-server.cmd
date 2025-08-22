@echo off
setlocal ENABLEDELAYEDEXPANSION
rem Tiny local HTTP server for EV Map (Windows .cmd)
set PORT=8000

rem Change to the script's directory
cd /d "%~dp0"

rem Try Python (python or py), then Node (npx http-server)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Starting Python HTTP server on http://localhost:%PORT%/
  start "" "http://localhost:%PORT%/north_america_ev_map_offline.html"
  python -m http.server %PORT%
  goto :eof
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Starting Python HTTP server on http://localhost:%PORT%/
  start "" "http://localhost:%PORT%/north_america_ev_map_offline.html"
  py -m http.server %PORT%
  goto :eof
)

where npx >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Starting Node http-server on http://localhost:%PORT%/
  start "" "http://localhost:%PORT%/north_america_ev_map_offline.html"
  npx --yes http-server -p %PORT%
  goto :eof
)

echo Could not find Python or Node.js on PATH.
echo Install Python from https://www.python.org/downloads/ or Node.js from https://nodejs.org/
pause
