llvm-symbolizer - convert addresses into source code locations
==============================================================

SYNOPSIS
--------

:program:`llvm-symbolizer` [options]

DESCRIPTION
-----------

:program:`llvm-symbolizer` reads object file names and addresses from standard
input and prints corresponding source code locations to standard output. This
program uses debug info sections and symbol table in the object files.

EXAMPLE
--------

.. code-block:: console

  $ cat addr.txt
  a.out 0x4004f4
  /tmp/b.out 0x400528
  /tmp/c.so 0x710
  $ llvm-symbolizer < addr.txt
  main
  /tmp/a.cc:4
  
  f(int, int)
  /tmp/b.cc:11

  h_inlined_into_g
  /tmp/header.h:2
  g_inlined_into_f
  /tmp/header.h:7
  f_inlined_into_main
  /tmp/source.cc:3
  main
  /tmp/source.cc:8

OPTIONS
-------

.. option:: -functions

  Print function names as well as source file/line locations. Defaults to true.

.. option:: -use-symbol-table

 Prefer function names stored in symbol table to function names
 in debug info sections. Defaults to true.

.. option:: -demangle

 Print demangled function names. Defaults to true.

.. option:: -inlining 

 If a source code location is in an inlined function, prints all the
 inlnied frames. Defaults to true.

EXIT STATUS
-----------

:program:`llvm-symbolizer` returns 0. Other exit codes imply internal program error.
