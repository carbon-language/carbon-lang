@echo off

echo Installing MSVC integration...
set SUCCESS=0

REM Change to the directory of this batch file.
cd /d %~dp0

REM Search for the MSBuild toolsets directory.
SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V100
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V100

:TRY_V110
SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V110
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V110

:TRY_V120
SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\V120\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V120
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\V120\Platforms\Win32\PlatformToolsets"
IF EXIST %D% GOTO FOUND_V120

IF %SUCCESS% == 1 goto DONE
echo Failed to find MSBuild toolsets directory.
goto FAILED


:FOUND_V100
IF NOT EXIST %D%\LLVM-vs2010 mkdir %D%\LLVM-vs2010
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2010.props %D%\LLVM-vs2010
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2010.targets %D%\LLVM-vs2010
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
set SUCCESS=1
GOTO TRY_V110

:FOUND_V110
IF NOT EXIST %D%\LLVM-vs2012 mkdir %D%\LLVM-vs2012
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2012.props %D%\LLVM-vs2012
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2012.targets %D%\LLVM-vs2012
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
IF NOT EXIST %D%\LLVM-vs2012_xp mkdir %D%\LLVM-vs2012_xp
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2012_xp.props %D%\LLVM-vs2012_xp
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy Microsoft.Cpp.Win32.LLVM-vs2012_xp.targets %D%\LLVM-vs2012_xp
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
set SUCCESS=1
GOTO TRY_V120

:FOUND_V120
IF NOT EXIST %D%\LLVM-vs2013 mkdir %D%\LLVM-vs2013
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy toolset-vs2013.props %D%\LLVM-vs2013\toolset.props
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy toolset-vs2013.targets %D%\LLVM-vs2013\toolset.targets
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
IF NOT EXIST %D%\LLVM-vs2013_xp mkdir %D%\LLVM-vs2013_xp
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy toolset-vs2013_xp.props %D%\LLVM-vs2013_xp\toolset.props
IF NOT %ERRORLEVEL% == 0 GOTO FAILED
copy toolset-vs2013_xp.targets %D%\LLVM-vs2013_xp\toolset.targets
IF NOT %ERRORLEVEL% == 0 GOTO FAILED

:DONE
echo Done!
goto END

:FAILED
echo MSVC integration install failed.
pause
goto END

:END
