# Setup Windows Task Scheduler for Daily Updates
# Run this script as Administrator

$TaskName = "StockPredictor-DailyUpdate"
$ScriptPath = "$PSScriptRoot\run_daily_update.bat"
$TaskTime = "06:00"  # 6 AM daily

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create task action
$Action = New-ScheduledTaskAction -Execute $ScriptPath

# Create task trigger (daily at 6 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At $TaskTime

# Create task settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily stock data update for Stock Predictor"

Write-Host "`n✅ Task scheduled successfully!" -ForegroundColor Green
Write-Host "Task Name: $TaskName" -ForegroundColor Cyan
Write-Host "Schedule: Daily at $TaskTime" -ForegroundColor Cyan
Write-Host "Script: $ScriptPath" -ForegroundColor Cyan
Write-Host "`nTo view/modify: Open Task Scheduler (taskschd.msc) and look for '$TaskName'" -ForegroundColor Yellow
