
			ZLib for Ada thick binding (ZLib.Ada)
			Release 1.2

ZLib.Ada is a thick binding interface to the popular ZLib data
compression library, available at http://www.gzip.org/zlib/.
It provides Ada-style access to the ZLib C library.


	Here are the main changes since ZLib.Ada 1.1:

- The default header type has a name "Default" now. Auto is used only for
  automatic GZip/ZLib header detection.

- Added test for multitasking mtest.adb.

- Added GNAT project file zlib.gpr.


	How to build ZLib.Ada under GNAT

You should have the ZLib library already build on your computer, before
building ZLib.Ada. Make the directory of ZLib.Ada sources current and
issue the command:

  gnatmake test -largs -L<directory where libz.a is> -lz

Or use the GNAT project file build for GNAT 3.15 or later:

  gnatmake -Pzlib.gpr -L<directory where libz.a is>


	How to build ZLib.Ada under Aonix ObjectAda for Win32 7.2.2

1. Make a project with all *.ads and *.adb files from the distribution.
2. Build the libz.a library from the ZLib C sources.
3. Rename libz.a to z.lib.
4. Add the library z.lib to the project.
5. Add the libc.lib library from the ObjectAda distribution to the project.
6. Build the executable using test.adb as a main procedure.


	How to use ZLib.Ada

The source files test.adb and read.adb are small demo programs that show
the main functionality of ZLib.Ada.

The routines from the package specifications are commented.


Homepage: http://zlib-ada.sourceforge.net/
Author: Dmitriy Anisimkov <anisimkov@yahoo.com>
