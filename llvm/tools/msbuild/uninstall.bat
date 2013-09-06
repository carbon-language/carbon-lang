@echo off

echo Uninstalling MSVC integration...

REM CD to the directory of this batch file.
cd /d %~dp0

SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.Win32.LLVM-vs2010.props
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.Win32.LLVM-vs2010.targets
IF EXIST %D%\LLVM-vs2010 rmdir %D%\LLVM-vs2010

SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\Platforms\Win32\PlatformToolsets"
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.Win32.LLVM-vs2010.props
IF EXIST %D%\LLVM-vs2010 del %D%\LLVM-vs2010\Microsoft.Cpp.Win32.LLVM-vs2010.targets
IF EXIST %D%\LLVM-vs2010 rmdir %D%\LLVM-vs2010

SET D="%ProgramFiles%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\Win32\PlatformToolsets"
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.Win32.LLVM-vs2012.props
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.Win32.LLVM-vs2012.targets
IF EXIST %D%\LLVM-vs2012 rmdir %D%\LLVM-vs2012

SET D="%ProgramFiles(x86)%\MSBuild\Microsoft.Cpp\v4.0\V110\Platforms\Win32\PlatformToolsets"
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.Win32.LLVM-vs2012.props
IF EXIST %D%\LLVM-vs2012 del %D%\LLVM-vs2012\Microsoft.Cpp.Win32.LLVM-vs2012.targets
IF EXIST %D%\LLVM-vs2012 rmdir %D%\LLVM-vs2012

echo Done!
