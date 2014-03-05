.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

===============
Windows support
===============

LLD has some experimental Windows support. When invoked as ``link.exe`` or with
``-flavor link``, the driver for Windows operating system is used to parse
command line options, and it drives further linking processes. LLD accepts
almost all command line options that the linker shipped with Microsoft Visual
C++ (link.exe) supports.

The current status is that LLD can link itself on Windows x86 using Visual C++
as the compiler.

Development status
==================

Driver
  :good:`Mostly done`. Some exotic command line options that are not usually
  used for application develompent, such as ``/DRIVER``, are not supported.
  Options for Windows 8 app store are not recognized too
  (e.g. ``/APPCONTAINER``).

Linking against DLL
  :good:`Done`. LLD can read import libraries needed to link against DLL. Both
  export-by-name and export-by-ordinal are supported.

Linking against static library
  :good:`Done`. The format of static library (.lib) on Windows is actually the
  same as on Unix (.a). LLD can read it.

Creating DLL
  :good:`Done`. LLD creates a DLL if ``/DLL`` option is given. Exported
  functions can be specified either via command line (``/EXPORT``) or via
  module-definition file (.def). Both export-by-name and export-by-ordinal are
  supported. LLD uses Microsoft ``lib.exe`` tool to create an import library
  file.

Windows resource files support
  :good:`Done`. If an ``.rc`` file is given, LLD converts the file to a COFF
  file using some external commands and link it. Specifically, ``rc.exe`` is
  used to compile a resource file (.rc) to a compiled resource (.res)
  file. ``rescvt.exe`` is then used to convert a compiled resource file to a
  COFF object file section. Both tools are shipped with MSVC.

Safe Structured Exception Handler (SEH)
  :good:`Done` for x86. :partial:`Work in progress` for x64.

Module-definition file
  :partial:`Partially done`. LLD currently recognizes these directives:
  ``EXPORTS``, ``HEAPSIZE``, ``STACKSIZE``, ``NAME``, and ``VERSION``.

x64 (x86-64)
  :partial:`Work in progress`. LLD can create PE32+ executable but the generated
  file does not work unless source object files are very simple because of the
  lack of SEH handler table.

Debug info
  :none:`No progress has been made`. Microsoft linker can interpret the CodeGen
  debug info (old-style debug info) and PDB to emit an .pdb file. LLD doesn't
  support neither.


Known issues
============

Note that LLD is still in early stage in development, so there are still many
bugs. Here is a list of notable bugs.

* Symbol name resolution from library files sometimes fails. On Windows, the
  order of library files in command line does not matter, but LLD sometimes
  fails to simulate the semantics. A workaround for it is to explicitly add
  library files to command line with ``/DEFAULTLIB``.

* Subsystem inference is not very reliable. Linker is supposed to set
  ``subsystem`` field in the PE/COFF header according to entry function name,
  but LLD sometimes ended up with ``unknown`` subsystem type. You need to give
  ``/SUBSYSTEM`` option if it fails to infer it.
