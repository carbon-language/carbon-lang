Microsoft Developer Studio Build File, Format Version 6.00 for
libpng 1.2.5 (October 3, 2002) and zlib

Copyright (C) 2000 Simon-Pierre Cadieux
For conditions of distribution and use, see copyright notice in png.h

Assumes that libpng sources are in ..\..
Assumes that zlib sources have been copied to ..\..\..\zlib

To build:

0) On the main menu, select "File | Open Workspace" and then
   select "libpng.dsw".

1) On the main menu Select "Build | Set Active configuration". 
   Among the configurations beginning with "libpng" select the 
   one you wish to build (the corresponding "zlib" configuration
   will be built automatically).

2) Select "Build | Clean"

3) Select "Build | Rebuild All".  Ignore warning messages about
   not being able to find certain include files (e.g., m68881.h,
   alloc.h).

4) Look in the appropriate "win32" subdirectories for both "zlib"
   and "libpng" binaries.

This project will build the PNG Development Group's "official" versions of
libpng and zlib libraries:

   libpng13.dll          (default version, currently C code only)
   libpng13a.dll         (C + Assembler version)
   libpng13b.dll         (C + Assembler debug version)
   libpng13d.dll         (C code debug version)
   libpng13vb.dll       (version for VB, uses "stdcall" protocol)
   libpng13[c,e-m].dll   (reserved for official versions) 
   libpng13[n-z].dll     (available for private versions)
   zlib.dll             (default version, currently C code only)
   zlibd.dll            (debug version)
   zlibvb.dll           (version for Visual Basic, uses "stdcall" protocol)

If you change anything in libpng, or select different compiler settings,
please change the library name to an unreserved name, and define
DLLFNAME_POSTFIX and (PRIVATEBUILD or SPECIALBUILD) accordingly. DLLFNAME_POSTFIX
should correspond to a string in the range of "N" to "Z" depending on the letter 
you choose for your private version.

All DLLs built by this project use the Microsoft dynamic C runtime library
MSVCRT.DLL (MSVCRTD.DLL for debug versions). If you distribute any of the
above mentioned libraries you should also include this DLL in your package.
For a list of files that are redistributable in Visual C++ 6.0, see
Common\Redist\Redist.txt on Disc 1 of the Visual C++ 6.0 product CDs. 

5) For an example workspace that builds an application using the resulting
   DLLs, go to Libpng's contrib\msvctest directory and use it to build
   and run "pngtest".
