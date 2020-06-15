@echo off
echo "This computer will be lock within 15 seconds, Please wear a mask within 5 seconds."
choice /c ac /n /t 5 /d a
if %errorlevel%==1 goto :exit
if %errorlevel%==2 goto :CONTINUE

:exit

timeout /t 10 /nobreak

:Continue
rundll32.exe user32.dll, LockWorkStation
