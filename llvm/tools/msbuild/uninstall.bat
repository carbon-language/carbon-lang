@echo off

echo Uninstalling MSVC integration...

REM CD to the directory of this batch file.
cd /d %~dp0

set PLATFORM=None
:START
IF %PLATFORM% == x64 GOTO END
IF %PLATFORM% == Win32 SET PLATFORM=x64
IF %PLATFORM% == None SET PLATFORM=Win32


SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.%PLATFORM%.LLVM-vs2010.props
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.%PLATFORM%.LLVM-vs2010.targets
IF EXIST %D%\LLVM-vs2010 rmdir %D%\LLVM-vs2010
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.%PLATFORM%.LLVM-vs2010.props
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.%PLATFORM%.LLVM-vs2010.targets
IF EXIST %D%\LLVM-vs2010 rmdir %D%\LLVM-vs2010

SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012.props
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012.targets
IF EXIST %D%\LLVM-vs2012 rmdir %D%\LLVM-vs2012
IF EXIST %D%\LLVM-vs2012_xp del %D%\LLVM-vs2012_xp\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012_xp.props
IF EXIST %D%\LLVM-vs2012_xp del %D%\LLVM-vs2012_xp\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012_xp.targets
IF EXIST %D%\LLVM-vs2012_xp rmdir %D%\LLVM-vs2012_xp
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012.props
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012.targets
IF EXIST %D%\LLVM-vs2012 rmdir %D%\LLVM-vs2012
IF EXIST %D%\LLVM-vs2012_xp del %D%\LLVM-vs2012_xp\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012_xp.props
IF EXIST %D%\LLVM-vs2012_xp del %D%\LLVM-vs2012_xp\Microsoft.Cpp.%PLATFORM%.LLVM-vs2012_xp.targets
IF EXIST %D%\LLVM-vs2012_xp rmdir %D%\LLVM-vs2012_xp

SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\V120\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2013 del %D%\LLVM-vs2013\toolset.props
IF EXIST %D%\LLVM-vs2013 del %D%\LLVM-vs2013\toolset.targets
IF EXIST %D%\LLVM-vs2013 rmdir %D%\LLVM-vs2013
IF EXIST %D%\LLVM-vs2013_xp del %D%\LLVM-vs2013_xp\toolset.props
IF EXIST %D%\LLVM-vs2013_xp del %D%\LLVM-vs2013_xp\toolset.targets
IF EXIST %D%\LLVM-vs2013_xp rmdir %D%\LLVM-vs2013_xp
SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\V120\Platforms\%PLATFORM%\PlatformToolsets"
IF EXIST %D%\LLVM-vs2013 del %D%\LLVM-vs2013\toolset.props
IF EXIST %D%\LLVM-vs2013 del %D%\LLVM-vs2013\toolset.targets
IF EXIST %D%\LLVM-vs2013 rmdir %D%\LLVM-vs2013
IF EXIST %D%\LLVM-vs2013_xp del %D%\LLVM-vs2013_xp\toolset.props
IF EXIST %D%\LLVM-vs2013_xp del %D%\LLVM-vs2013_xp\toolset.targets
IF EXIST %D%\LLVM-vs2013_xp rmdir %D%\LLVM-vs2013_xp


GOTO START

:END
echo Done!
