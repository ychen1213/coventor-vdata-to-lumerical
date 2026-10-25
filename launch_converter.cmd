@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "VDATA_GUI_PYTHON="

rem Explicit configuration takes priority and must point to an executable.
if defined VDATA_CONVERTER_PYTHON (
    if not exist "%VDATA_CONVERTER_PYTHON%" (
        echo VDATA_CONVERTER_PYTHON does not exist: "%VDATA_CONVERTER_PYTHON%"
        pause
        exit /b 1
    )
    set "VDATA_GUI_PYTHON=%VDATA_CONVERTER_PYTHON%"
    goto launch
)

rem Prefer a system Python with tkinter; it needs no scientific packages.
for /f "delims=" %%P in ('py -3 -c "import sys, tkinter; assert sys.version_info >= (3,6); print(sys.executable)" 2^>nul') do set "VDATA_GUI_PYTHON=%%P"
if defined VDATA_GUI_PYTHON goto launch
for /f "delims=" %%P in ('python -c "import sys, tkinter; assert sys.version_info >= (3,6); print(sys.executable)" 2^>nul') do set "VDATA_GUI_PYTHON=%%P"
if defined VDATA_GUI_PYTHON goto launch

rem Windows PowerShell only locates an installed Ansys interpreter. It does
rem not load the SDK or read a license. Search local fixed drives, not shares.
for /f "delims=" %%P in ('powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; $roots=@($env:LUMERICAL_ROOT,$env:LUMERICAL_INSTALL_ROOT); Get-ChildItem Env: | Where-Object Name -Match '^AWP_ROOT[0-9]+$' | ForEach-Object { $roots+=Join-Path $_.Value 'Lumerical' }; $bases=@($env:ProgramFiles,${env:ProgramFiles(x86)},$env:ProgramW6432); Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object { $bases+=$_.DeviceID+'\'; $bases+=$_.DeviceID+'\Program Files'; $bases+=$_.DeviceID+'\Program Files (x86)' }; foreach($base in ($bases | Where-Object { $_ } | Select-Object -Unique)) { foreach($pattern in @('Ansys Inc\v*\Lumerical','Ansys\v*\Lumerical','Lumerical\v*','Lumerical')) { Get-ChildItem -Path (Join-Path $base $pattern) -Directory | ForEach-Object { $roots+=$_.FullName } } }; $found=@(); foreach($root in ($roots | Where-Object { $_ } | Select-Object -Unique)) { $found+=Get-ChildItem -Path (Join-Path $root 'python*\pythonw.exe') -File }; foreach($exe in ($found | Sort-Object { [regex]::Replace($_.FullName,'[0-9]+',{param($m) $m.Value.PadLeft(12,'0')}) } -Descending)) { $probe=Join-Path $exe.Directory.FullName 'python.exe'; if(Test-Path -LiteralPath $probe) { & $probe -c 'import sys, tkinter; assert sys.version_info >= (3,6)' 2>$null; if($LASTEXITCODE -eq 0) { $exe.FullName; break } } }"') do set "VDATA_GUI_PYTHON=%%P"
if defined VDATA_GUI_PYTHON goto launch

echo Python 3.6 or newer with tkinter was not found.
echo Install Python 3, or set VDATA_CONVERTER_PYTHON to pythonw.exe or python.exe.
pause
exit /b 1

:launch
rem Use the windowless sibling when available, keeping the selected runtime.
for %%I in ("%VDATA_GUI_PYTHON%") do if exist "%%~dpIpythonw.exe" set "VDATA_GUI_PYTHON=%%~dpIpythonw.exe"
start "" "%VDATA_GUI_PYTHON%" "%~dp0gui.py" %*
exit /b 0
