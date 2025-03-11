@echo off

REM プロジェクトをビルド
dotnet build --project tagmane.csproj

REM 実行可能ファイルを実行
dotnet run --project tagmane.csproj

pause
