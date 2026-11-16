@echo off
rem Dream Game Finder - double-click to start. Close this window to stop.
cd /d "%~dp0"
title Dream Game Finder

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" during setup^), then run this again.
  pause
  exit /b 1
)

python -c "import numpy, scipy" >nul 2>nul
if errorlevel 1 (
  echo Installing numpy and scipy ^(one time only^)...
  python -m pip install --user numpy scipy
  if errorlevel 1 (
    echo Could not install numpy/scipy. Try: python -m pip install numpy scipy
    pause
    exit /b 1
  )
)

if not exist recsys_model.npz (
  if not exist steam.db (
    echo No game data yet. Collect it first ^(see README.md^):
    echo   python steam_reviews.py games
    echo   python steam_reviews.py reviews --limit-games 3000 --max-per-game 300
    pause
    exit /b 1
  )
  echo Building the recommender ^(about half a minute^)...
  python recommend.py build
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

echo.
echo Starting Dream Game Finder - your browser will open in a moment.
echo Keep this window open while you use the site. Close it to stop.
echo.
python server.py --open --port 8000 %*
pause
