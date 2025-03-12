@echo off

call cuda-version-check.bat

echo Loading tagmane ...

dotnet run --project tagmane.csproj

echo tagmane Completed.

pause
