Microsoft Developer Studio Build File, Format Version 6.00 for
msvctest

Assumes that libpng DLLs and LIBs are in ..\..\projects\msvc\win32\libpng
Assumes that zlib DLLs and LIBs are in ..\..\projects\msvc\win32\zlib

To build:

1) On the main menu Select "Build|Set Active configuration".
   Choose the configuration that corresponds to the library you want to test.
   This library must have been built using the libpng MS project located in
   the "mscv" subdirectory.

2) Select "Build|Clean"

3) Select "Build|Rebuild All"

4) The test results should appear in the "Build" pane of the Output Window.


Simon-Pierre Cadieux
Methodex Computer Systems Inc.
