# Microsoft Developer Studio Project File - Name="VisualPng" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00

# Copyright 2000, Willem van Schaik.  For conditions of distribution and
# use, see the copyright/license/disclaimer notice in png.h

# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=VisualPng - Win32 Debug LIB
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "VisualPng.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "VisualPng.mak" CFG="VisualPng - Win32 Debug LIB"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "VisualPng - Win32 DLL" (based on "Win32 (x86) Application")
!MESSAGE "VisualPng - Win32 Debug DLL" (based on "Win32 (x86) Application")
!MESSAGE "VisualPng - Win32 LIB" (based on "Win32 (x86) Application")
!MESSAGE "VisualPng - Win32 Debug LIB" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "VisualPng - Win32 DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "VisualPng___Win32_DLL"
# PROP BASE Intermediate_Dir "VisualPng___Win32_DLL"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "dll"
# PROP Intermediate_Dir "dll"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /I "libpng" /I "zlib" /D "PNG_USE_DLL" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_USE_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 libpng13.lib zlibd.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386 /libpath:"libpng" /libpath:"zlib"
# ADD LINK32 libpng13.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386 /libpath:"..\..\projects\msvc\win32\libpng\dll"
# Begin Special Build Tool
OutDir=.\dll
SOURCE="$(InputPath)"
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll;..\..\projects\msvc\win32\zlib\dll;	$(outdir)\VisualPng.exe ..\..\contrib\pngsuite\basn6a16.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "VisualPng - Win32 Debug DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "VisualPng___Win32_Debug_DLL"
# PROP BASE Intermediate_Dir "VisualPng___Win32_Debug_DLL"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "dll_dbg"
# PROP Intermediate_Dir "dll_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /I "libpng" /I "zlib" /D "PNG_USE_DLL" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_USE_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 libpng13.lib zlibd.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libc" /pdbtype:sept /libpath:"libpng" /libpath:"zlib"
# SUBTRACT BASE LINK32 /nodefaultlib
# ADD LINK32 libpng13d.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libc" /pdbtype:sept /libpath:"..\..\projects\msvc\win32\libpng\dll_dbg"
# SUBTRACT LINK32 /nodefaultlib
# Begin Special Build Tool
OutDir=.\dll_dbg
SOURCE="$(InputPath)"
PostBuild_Cmds=set path=..\..\projects\msvc\win32\libpng\dll_dbg;..\..\projects\msvc\win32\zlib\dll_dbg;	$(outdir)\VisualPng.exe ..\..\contrib\pngsuite\basn6a16.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "VisualPng - Win32 LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "VisualPng___Win32_LIB"
# PROP BASE Intermediate_Dir "VisualPng___Win32_LIB"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "lib"
# PROP Intermediate_Dir "lib"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_USE_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /c
# SUBTRACT BASE CPP /YX
# ADD CPP /nologo /W3 /GX /O2 /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 libpng13.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386 /libpath:"..\..\projects\msvc\win32\libpng\dll"
# ADD LINK32 libpng.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386 /libpath:"..\..\projects\msvc\win32\libpng\lib"
# Begin Special Build Tool
OutDir=.\lib
SOURCE="$(InputPath)"
PostBuild_Cmds=$(outdir)\VisualPng.exe ..\..\contrib\pngsuite\basn6a16.png
# End Special Build Tool

!ELSEIF  "$(CFG)" == "VisualPng - Win32 Debug LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "VisualPng___Win32_Debug_LIB"
# PROP BASE Intermediate_Dir "VisualPng___Win32_Debug_LIB"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "lib_dbg"
# PROP Intermediate_Dir "lib_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_USE_DLL" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "..\..\..\libpng" /I "..\..\..\zlib" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "PNG_NO_STDIO" /D "PNG_NO_GLOBAL_ARRAYS" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 libpng13d.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libc" /pdbtype:sept /libpath:"..\..\projects\msvc\win32\libpng\dll_dbg"
# SUBTRACT BASE LINK32 /nodefaultlib
# ADD LINK32 libpng.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libc" /pdbtype:sept /libpath:"..\..\projects\msvc\win32\libpng\lib_dbg"
# SUBTRACT LINK32 /nodefaultlib
# Begin Special Build Tool
OutDir=.\lib_dbg
SOURCE="$(InputPath)"
PostBuild_Cmds=$(outdir)\VisualPng.exe ..\..\contrib\pngsuite\basn6a16.png
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "VisualPng - Win32 DLL"
# Name "VisualPng - Win32 Debug DLL"
# Name "VisualPng - Win32 LIB"
# Name "VisualPng - Win32 Debug LIB"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\PngFile.c
# End Source File
# Begin Source File

SOURCE=.\VisualPng.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\cexcept.h
# End Source File
# Begin Source File

SOURCE=.\PngFile.h
# End Source File
# Begin Source File

SOURCE=.\resource.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\VisualPng.ico
# End Source File
# Begin Source File

SOURCE=.\VisualPng.rc
# End Source File
# End Group
# End Target
# End Project
