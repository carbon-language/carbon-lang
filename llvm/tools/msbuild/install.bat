@echo off

echo Installing MSVC integration...

REM Change to the directory of this batch file.
cd /d %~dp0

REM Search for the MSBuild toolsets directory.
SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_MSBUILD
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_MSBUILD

echo Failed to find MSBuild toolsets directory.
goto FAILED

:FOUND_MSBUILD
IF NOT EXIST %D%\llvm mkdir %D%\llvm
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

copy Microsoft.Cpp.Win32.llvm.props %D%\llvm
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.llvm.targets %D%\llvm
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

echo Done!
goto END

:FAILED
echo MSVC integration install failed.
pause
goto END

:END
