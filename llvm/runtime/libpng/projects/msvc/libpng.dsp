# Microsoft Developer Studio Project File - Name="libpng" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102
# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=libpng - Win32 DLL
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "libpng.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libpng.mak" CFG="libpng - Win32 DLL"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libpng - Win32 DLL" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libpng - Win32 DLL Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libpng - Win32 DLL ASM" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libpng - Win32 DLL Debug ASM" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "libpng - Win32 LIB" (based on "Win32 (x86) Static Library")
!MESSAGE "libpng - Win32 LIB Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "libpng - Win32 DLL VB" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""

!IF  "$(CFG)" == "libpng - Win32 DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\libpng\dll"
# PROP Intermediate_Dir ".\win32\libpng\dll"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /I "..\.." /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /Yu"png.h" /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386
# ADD LINK32 /nologo /dll /machine:I386 /out:".\win32\libpng\dll\libpng13.dll"
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\libpng\dll_dbg"
# PROP Intermediate_Dir ".\win32\libpng\dll_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "..\.." /I "..\..\..\zlib" /D "DEBUG" /D "_DEBUG" /D PNG_DEBUG=1 /D "WIN32" /D "_WINDOWS" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /Yu"png.h" /FD /GZ /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "_DEBUG" /d PNG_DEBUG=1
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 /nologo /dll /debug /machine:I386 /out:".\win32\libpng\dll_dbg\libpng13d.dll"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL ASM"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\libpng\dll_asm"
# PROP Intermediate_Dir ".\win32\libpng\dll_asm"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /I "..\.." /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "PNG_USE_PNGVCRD" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /Yu"png.h" /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "NDEBUG" /d "PNG_USE_PNGVCRD"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386
# ADD LINK32 /nologo /dll /machine:I386 /out:".\win32\libpng\dll_asm\libpng13a.dll"
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug ASM"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\libpng\dll_dbga"
# PROP Intermediate_Dir ".\win32\libpng\dll_dbga"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "..\.." /I "..\..\..\zlib" /D "DEBUG" /D "_DEBUG" /D PNG_DEBUG=1 /D "WIN32" /D "_WINDOWS" /D "PNG_USE_PNGVCRD" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /Yu"png.h" /FD /GZ /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "_DEBUG" /d PNG_DEBUG=1 /d "PNG_USE_PNGVCRD"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 /nologo /dll /debug /machine:I386 /out:".\win32\libpng\dll_dbga\libpng13b.dll"

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\libpng\lib"
# PROP Intermediate_Dir ".\win32\libpng\lib"
# PROP Target_Dir ""
MTL=midl.exe
CPP=cl.exe
# ADD BASE CPP /nologo /W3 /O1 /D "WIN32" /D "NDEBUG" /D "_LIB" /FD /c
# ADD CPP /nologo /W3 /O1 /I "..\.." /I "..\..\..\zlib" /D "WIN32" /D "NDEBUG" /Yu"png.h" /FD /c
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\libpng\lib_dbg"
# PROP Intermediate_Dir ".\win32\libpng\lib_dbg"
# PROP Target_Dir ""
MTL=midl.exe
CPP=cl.exe
# ADD BASE CPP /nologo /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_LIB" /FD /GZ /c
# ADD CPP /nologo /W3 /Zi /Od /I "..\.." /I "..\..\..\zlib" /D "DEBUG" /D "_DEBUG" /D PNG_DEBUG=1 /D "WIN32" /Yu"png.h" /FD /GZ /c
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL VB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "libpng___Win32_DLL_VB"
# PROP BASE Intermediate_Dir "libpng___Win32_DLL_VB"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\libpng\dll_vb"
# PROP Intermediate_Dir ".\win32\libpng\dll_vb"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /I "..\.." /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /Yu"png.h" /FD /c
# ADD CPP /nologo /Gd /MD /W3 /O1 /I "..\.." /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "PNG_BUILD_DLL" /D "ZLIB_DLL" /D PNGAPI=__stdcall /Yu"png.h" /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /i "..\.." /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386 /out:".\win32\libpng\dll\libpng13.dll"
# SUBTRACT BASE LINK32 /pdb:none
# ADD LINK32 /nologo /dll /machine:I386 /out:".\win32\libpng\dll_vb\libpngvb13.dll"
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "libpng - Win32 DLL"
# Name "libpng - Win32 DLL Debug"
# Name "libpng - Win32 DLL ASM"
# Name "libpng - Win32 DLL Debug ASM"
# Name "libpng - Win32 LIB"
# Name "libpng - Win32 LIB Debug"
# Name "libpng - Win32 DLL VB"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\png.c
# SUBTRACT CPP /YX /Yc /Yu
# End Source File
# Begin Source File

SOURCE=.\png.rc

!IF  "$(CFG)" == "libpng - Win32 DLL"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL VB"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\png32ms.def

!IF  "$(CFG)" == "libpng - Win32 DLL"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL VB"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\pngerror.c
# ADD CPP /Yc"png.h"
# End Source File
# Begin Source File

SOURCE=..\..\pngget.c
# End Source File
# Begin Source File

SOURCE=..\..\pngmem.c
# End Source File
# Begin Source File

SOURCE=..\..\pngpread.c
# End Source File
# Begin Source File

SOURCE=..\..\pngread.c
# End Source File
# Begin Source File

SOURCE=..\..\pngrio.c
# End Source File
# Begin Source File

SOURCE=..\..\pngrtran.c
# End Source File
# Begin Source File

SOURCE=..\..\pngrutil.c
# End Source File
# Begin Source File

SOURCE=..\..\pngset.c
# End Source File
# Begin Source File

SOURCE=..\..\pngtrans.c
# End Source File
# Begin Source File

SOURCE=..\..\pngvcrd.c

!IF  "$(CFG)" == "libpng - Win32 DLL"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL VB"

# PROP BASE Exclude_From_Build 1
# PROP Exclude_From_Build 1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\pngwio.c
# End Source File
# Begin Source File

SOURCE=..\..\pngwrite.c
# End Source File
# Begin Source File

SOURCE=..\..\pngwtran.c
# End Source File
# Begin Source File

SOURCE=..\..\pngwutil.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\png.h
# End Source File
# Begin Source File

SOURCE=..\..\pngconf.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Source File

SOURCE=.\readme.txt

!IF  "$(CFG)" == "libpng - Win32 DLL"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL ASM"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL Debug ASM"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "libpng - Win32 DLL VB"

# PROP BASE Exclude_From_Build 1
# PROP Exclude_From_Build 1

!ENDIF 

# End Source File
# End Target
# End Project
