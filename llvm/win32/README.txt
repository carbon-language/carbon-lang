Directory structure
===================
Although I have made every effort not to use absolute paths, I have only tested building
with my own directory structure and it looks like this:

c:\project\llvm                  ; Main project directory
c:\project\llvm\win32            ; win32 project
c:\project\llvm\win32\tools      ; flex, bison and sed live here
c:\project\llvm\win32\share      ; flex, bison and sed support files

Requirements
============

You need flex, sed and bison - I'm using the GnuWin32 versions of these tools which can be obtained from

http://gnuwin32.sourceforge.net/

Limitations
============

At the moment only the core LLVM libraries and the tablegen executable are built. If anyone has time to
port the rest of the LLVM tools it would be great...

Other notes
===========

When linking with your own application it is of the utmost importance that you use the same runtime
libraries in compiling LLVM as in your own project. Otherwise you will get a lot of errors. To change this,
just mark all the projects except the Config project (since it doesn't use the C compiler) in the
solution explorer, select properties - then go to the C/C++ options and the Code Generation sub option page.
In the Runtime Library (6th from the top) select the appropriate version. Then change the active
configuration to Release (in the top left corner of the properties window) and select the appropriate
runtime library for the release version.

When linking with your applications, you need to force a symbol reference to bring in the x86 backend.
Open the properties for your main project and select the Linker options - under the Input options there
is a Force Symbol References field where you need to enter _X86TargetMachineModule. If anyone has a better
suggestion for how to trick the linker into always pulling in these objects, I'd be grateful...

Contact Information
===================

please contact me at this address if you have any questions:

morten@hue.no


-- Morten Ofstad 2.11.2004
