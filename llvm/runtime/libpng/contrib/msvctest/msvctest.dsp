# Microsoft Developer Studio Project File - Name="msvctest" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=msvctest - Win32 Debug DLL
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "msvctest.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "msvctest.mak" CFG="msvctest - Win32 Debug DLL"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "msvctest - Win32 DLL" (based on "Win32 (x86) Console Application")
!MESSAGE "msvctest - Win32 Debug DLL" (based on "Win32 (x86) Console Application")
!MESSAGE "msvctest - Win32 ASM DLL" (based on "Win32 (x86) Console Application")
!MESSAGE "msvctest - Win32 Debug ASM DLL" (based on "Win32 (x86) Console Application")
!MESSAGE "msvctest - Win32 LIB" (based on "Win32 (x86) Console Application")
!MESSAGE "msvctest - Win32 Debug LIB" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "msvctest - Win32 DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "dll"
# PROP BASE Intermediate_Dir "dll"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "dll"
# PROP Intermediate_Dir "dll"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "PNG_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /machine:I386
# ADD LINK32 ..\..\projects\msvc\win32\libpng\dll\libpng13.lib /nologo /subsystem:console /machine:I386
# Begin Special Build Tool
OutDir=.\dll
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll;..\..\projects\msvc\win32\zlib\dll;	$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "msvctest - Win32 Debug DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "dll_dbg"
# PROP BASE Intermediate_Dir "dll_dbg"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "dll_dbg"
# PROP Intermediate_Dir "dll_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "..\..\..\zlib" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PNG_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ..\..\projects\msvc\win32\libpng\dll_dbg\libpng13d.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# Begin Special Build Tool
OutDir=.\dll_dbg
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll_dbg;..\..\projects\msvc\win32\zlib\dll_dbg;	$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "msvctest - Win32 ASM DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "dll_asm"
# PROP BASE Intermediate_Dir "dll_asm"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "dll_asm"
# PROP Intermediate_Dir "dll_asm"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "PNG_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /machine:I386
# ADD LINK32 ..\..\projects\msvc\win32\libpng\dll_asm\libpng13a.lib /nologo /subsystem:console /machine:I386
# Begin Special Build Tool
OutDir=.\dll_asm
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll_asm;..\..\projects\msvc\win32\zlib\dll_asm;	$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "msvctest - Win32 Debug ASM DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "dll_dbga"
# PROP BASE Intermediate_Dir "dll_dbga"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "dll_dbga"
# PROP Intermediate_Dir "dll_dbga"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "..\..\..\zlib" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PNG_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ..\..\projects\msvc\win32\libpng\dll_dbga\libpng13b.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# Begin Special Build Tool
OutDir=.\dll_dbga
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll_dbga;..\..\projects\msvc\win32\zlib\dll_dbga;	$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "msvctest - Win32 LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "lib"
# PROP BASE Intermediate_Dir "lib"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "lib"
# PROP Intermediate_Dir "lib"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /FD /c
# ADD CPP /nologo /W3 /O1 /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /machine:I386
# ADD LINK32 ..\..\projects\msvc\win32\libpng\lib\libpng.lib /nologo /subsystem:console /machine:I386
# Begin Special Build Tool
OutDir=.\lib
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "msvctest - Win32 Debug LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "lib_dbg"
# PROP BASE Intermediate_Dir "lib_dbg"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "lib_dbg"
# PROP Intermediate_Dir "lib_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Zi /Od /I "..\..\..\zlib" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ..\..\projects\msvc\win32\libpng\lib_dbg\libpng.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# Begin Special Build Tool
OutDir=.\lib_dbg
SOURCE="$(InputPath)"
PostBuild_Desc=[Run Test]
PostBuild_Cmds=$(outdir)\msvctest.exe ..\..\pngtest.png
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "msvctest - Win32 DLL"
# Name "msvctest - Win32 Debug DLL"
# Name "msvctest - Win32 ASM DLL"
# Name "msvctest - Win32 Debug ASM DLL"
# Name "msvctest - Win32 LIB"
# Name "msvctest - Win32 Debug LIB"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\pngtest.c
# End Source File
# Begin Source File

SOURCE=.\README.txt
# PROP Exclude_From_Build 1
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
