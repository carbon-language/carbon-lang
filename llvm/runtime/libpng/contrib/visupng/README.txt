Microsoft Developer Studio Build File, Format Version 6.00 for VisualPng
------------------------------------------------------------------------

Copyright 2000, Willem van Schaik.  For conditions of distribution and
use, see the copyright/license/disclaimer notice in png.h

As a PNG .dll demo VisualPng is finished. More features would only hinder
the program's objective. However, further extensions (like support for other 
graphics formats) are in development. To get these, or for pre-compiled 
binaries, go to "http://www.schaik.com/png/visualpng.html".

------------------------------------------------------------------------

Assumes that

   libpng DLLs and LIBs are in ..\..\projects\msvc\win32\libpng
   zlib DLLs and LIBs are in   ..\..\projects\msvc\win32\zlib
   libpng header files are in  ..\..\..\libpng
   zlib header files are in    ..\..\..\zlib
   the pngsuite images are in  ..\pngsuite

To build:

1) On the main menu Select "Build|Set Active configuration".
   Choose the configuration that corresponds to the library you want to test.
   This library must have been built using the libpng MS project located in
   the "..\..\mscv" subdirectory.

2) Select "Build|Clean"

3) Select "Build|Rebuild All"

4) After compiling and linking VisualPng will be started to view an image
   from the PngSuite directory.  Press Ctrl-N (and Ctrl-V) for other images.


To install:

When distributing VisualPng (or a further development) the following options
are available:

1) Build the program with the configuration "Win32 LIB" and you only need to
   include the executable from the ./lib directory in your distribution.

2) Build the program with the configuration "Win32 DLL" and you need to put
   in your distribution the executable from the ./dll directory and the dll's
   libpng1.dll, zlib.dll and msvcrt.dll.  These need to be in the user's PATH.


Willem van Schaik
Calgary, June 6th 2000

P.S. VisualPng was written based on preliminary work of:

    - Simon-Pierre Cadieux
    - Glenn Randers-Pehrson
    - Greg Roelofs

