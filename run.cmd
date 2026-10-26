@echo off
setlocal
if defined VDATA_STANDALONE_PYTHON goto ready
if exist "%~dp0.venv\Scripts\python.exe" set "VDATA_STANDALONE_PYTHON=%~dp0.venv\Scripts\python.exe"
if defined VDATA_STANDALONE_PYTHON goto ready
echo Set VDATA_STANDALONE_PYTHON to your ordinary Python executable.
echo Install requirements.txt with that Python before running this tool.
pause
exit /b 2
:ready
"%VDATA_STANDALONE_PYTHON%" "%~dp0run.py" %*
set "CONVERTER_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %CONVERTER_EXIT%
pause
exit /b %CONVERTER_EXIT%
