# Microsoft Developer Studio Project File - Name="zlib" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102
# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=zlib - Win32 DLL
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "zlib.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "zlib.mak" CFG="zlib - Win32 DLL"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "zlib - Win32 DLL" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "zlib - Win32 DLL Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "zlib - Win32 DLL ASM" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "zlib - Win32 DLL Debug ASM" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "zlib - Win32 LIB" (based on "Win32 (x86) Static Library")
!MESSAGE "zlib - Win32 LIB Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "zlib - Win32 DLL VB" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""

!IF  "$(CFG)" == "zlib - Win32 DLL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\zlib\dll"
# PROP Intermediate_Dir ".\win32\zlib\dll"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_USRDLL" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386
# ADD LINK32 /nologo /dll /machine:I386
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\zlib\dll_dbg"
# PROP Intermediate_Dir ".\win32\zlib\dll_dbg"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_USRDLL" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /D "DEBUG" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /FD /GZ /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 /nologo /dll /debug /machine:I386 /out:".\win32\zlib\dll_dbg\zlibd.dll"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL ASM"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\zlib\dll_asm"
# PROP Intermediate_Dir ".\win32\zlib\dll_asm"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /c
# ADD CPP /nologo /MD /W3 /O1 /I "..\..\..\zlib" /D "NDEBUG" /D "WIN32" /D "_WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /D "DYNAMIC_CRC_TABLE" /D "ASMV" /FAcs /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386
# ADD LINK32 gvmat32.obj /nologo /dll /machine:I386 /out:".\win32\zlib\dll_asm\zliba.dll"
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug ASM"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\zlib\dll_dbga"
# PROP Intermediate_Dir ".\win32\zlib\dll_dbga"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MDd /W3 /Zi /Od /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_USRDLL" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Zi /Od /I "..\..\..\zlib" /D "_DEBUG" /D "WIN32" /D "_WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /D "DYNAMIC_CRC_TABLE" /D "ASMV" /FAcs /FD /GZ /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /i "..\.." /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 gvmat32d.obj /nologo /dll /debug /machine:I386 /out:".\win32\zlib\dll_dbga\zlibb.dll"

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\zlib\lib"
# PROP Intermediate_Dir ".\win32\zlib\lib"
# PROP Target_Dir ""
MTL=midl.exe
CPP=cl.exe
# ADD BASE CPP /nologo /W3 /O1 /D "WIN32" /D "NDEBUG" /D "_LIB" /FD /c
# ADD CPP /nologo /W3 /O1 /D "WIN32" /D "NDEBUG" /FD /c
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\win32\zlib\lib_dbg"
# PROP Intermediate_Dir ".\win32\zlib\lib_dbg"
# PROP Target_Dir ""
MTL=midl.exe
CPP=cl.exe
# ADD BASE CPP /nologo /W3 /Zi /Od /D "WIN32" /D "_DEBUG" /D "_LIB" /FD /GZ /c
# ADD CPP /nologo /W3 /Zi /Od /D "WIN32" /D "_DEBUG" /FD /GZ /c
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL VB"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "zlib___Win32_DLL_VB"
# PROP BASE Intermediate_Dir "zlib___Win32_DLL_VB"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\win32\zlib\dll_vb"
# PROP Intermediate_Dir ".\win32\zlib\dll_vb"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
CPP=cl.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /FD /c
# ADD CPP /nologo /Gd /MD /W3 /O1 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "ZLIB_DLL" /FD /c
MTL=midl.exe
RSC=rc.exe
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /dll /machine:I386
# SUBTRACT BASE LINK32 /pdb:none
# ADD LINK32 /nologo /dll /machine:I386 /out:".\win32\zlib\dll_vb/zlibvb.dll"
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "zlib - Win32 DLL"
# Name "zlib - Win32 DLL Debug"
# Name "zlib - Win32 DLL ASM"
# Name "zlib - Win32 DLL Debug ASM"
# Name "zlib - Win32 LIB"
# Name "zlib - Win32 LIB Debug"
# Name "zlib - Win32 DLL VB"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\..\zlib\adler32.c
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\compress.c
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\crc32.c
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\deflate.c
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\contrib\asm386\gvmat32c.c

!IF  "$(CFG)" == "zlib - Win32 DLL"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL VB"

# PROP BASE Exclude_From_Build 1
# PROP Exclude_From_Build 1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\gzio.c
# ADD CPP /Yc"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infblock.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infcodes.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inffast.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inflate.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inftrees.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infutil.c
# ADD CPP /Yu"zutil.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\trees.c
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\uncompr.c
# End Source File
# Begin Source File

SOURCE=.\zlib.def

!IF  "$(CFG)" == "zlib - Win32 DLL"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL VB"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\msdos\zlib.rc

!IF  "$(CFG)" == "zlib - Win32 DLL"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL Debug ASM"

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 LIB Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "zlib - Win32 DLL VB"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\zutil.c
# ADD CPP /Yu"zutil.h"
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\..\zlib\deflate.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infblock.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infcodes.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inffast.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inffixed.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\inftrees.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\infutil.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\trees.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\zconf.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\zlib.h
# End Source File
# Begin Source File

SOURCE=..\..\..\zlib\zutil.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
