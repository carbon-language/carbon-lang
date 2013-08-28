@echo off

echo Uninstalling MSVC integration...

REM CD to the directory of this batch file.
cd /d %~dp0

REM Search for the MSBuild toolsets directory.
SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_MSBUILD
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_MSBUILD

echo Failed to find MSBuild toolsets directory.
goto FAILED

:FOUND_MSBUILD

del %D%\llvm\Microsoft.Cpp.Win32.llvm.props
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
del %D%\llvm\Microsoft.Cpp.Win32.llvm.targets
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
rmdir %D%\llvm
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

echo Done!
goto END

:FAILED
echo MSVC integration uninstall failed.
pause
goto END

:END
