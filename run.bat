@echo off
REM CVAT Statistics Tool
REM
REM Usage:
REM   run.bat annotations.xml
REM   run.bat annotations.xml --date 2026-05-11
REM   run.bat annotations.xml --date 2026-05-11 --revert
REM   run.bat annotations.xml --history 2026-05-10
REM   run.bat --list-history
REM
REM See: python main.py --help

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON=python3
) else (
    set PYTHON=py
)

%PYTHON% "%~dp0main.py" %*
pause
