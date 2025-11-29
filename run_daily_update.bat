@echo off
REM Daily Stock Data Update Script
REM Run this with Windows Task Scheduler

cd /d "%~dp0"

REM Activate virtual environment
call stock\Scripts\activate.bat

REM Run daily update
python daily_update.py

REM Deactivate
call stock\Scripts\deactivate.bat

echo Daily update completed at %date% %time%
