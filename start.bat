@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title InSAR Studio

set "PY="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem 1) Official Python launcher, if installed.
where py >nul 2>nul
if errorlevel 1 goto try_python
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto have_python

:try_python
rem 2) python on PATH.
where python >nul 2>nul
if errorlevel 1 goto try_common
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto have_python

:try_common
rem 3) Common install locations, without requiring PATH edits.
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if defined PY goto have_python
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto have_python
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PY goto have_python
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if defined PY goto have_python
if exist "%ProgramFiles%\Python313\python.exe" set "PY=%ProgramFiles%\Python313\python.exe"
if defined PY goto have_python
if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if defined PY goto have_python
if exist "%ProgramFiles%\Python311\python.exe" set "PY=%ProgramFiles%\Python311\python.exe"
if defined PY goto have_python
if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if defined PY goto have_python
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if defined PY goto have_python

echo.
echo [ERROR] Python 3.9+ was not found.
echo Install Python from https://www.python.org/downloads/
echo and enable "Add python.exe to PATH", then run start.bat again.
echo.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:have_python
echo Using Python: %PY%
"%PY%" "scripts\bootstrap.py" %*
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto finished
echo.
echo Startup failed with exit code %CODE%. See the messages above.
pause
:finished
exit /b %CODE%
