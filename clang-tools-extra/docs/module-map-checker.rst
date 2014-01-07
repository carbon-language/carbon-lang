.. index:: module-map-checker

================================
Module-Map-Checker User's Manual
================================

:program:`module-map-checker` is a tool that validates a module map by
checking that all headers in the corresponding directories are accounted for.

This program uses the Clang ModuleMap class to read and parse the module
map file.  Starting at the module map file directory, or just the include
paths, if specified, it will collect the names of all the files it
considers headers.  It then compares the headers against those referenced
in the module map, either explicitly named, or implicitly named via an
umbrella directory or umbrella file, as parsed by the ModuleMap object.
If headers are found which are not referenced or covered by an umbrella
directory or file, warning messages will be produced, and this program
will return an error code of 1.  Other errors result in an error code of 2.
If no problems are found, an error code of 0 is returned.

Note that in the case of umbrella headers, this tool invokes the compiler
to preprocess the file, and uses a callback to collect the header files
included by the umbrella header or any of its nested includes.  If any
front end options are needed for these compiler invocations, these
can be included on the command line after the module map file argument.

Warning message have the form::

  warning: module.map does not account for file: header.h

Note that for the case of the module map referencing a file that does
not exist, the module map parser in Clang will display an error message.

Getting Started
===============

To build from source:

1. Read `Getting Started with the LLVM System`_ and `Clang Tools
   Documentation`_ for information on getting sources for LLVM, Clang, and
   Clang Extra Tools.

2. `Getting Started with the LLVM System`_ and `Building LLVM with CMake`_ give
   directions for how to build. With sources all checked out into the
   right place the LLVM build will build Clang Extra Tools and their
   dependencies automatically.

   * If using CMake, you can also use the ``module-map-checker`` target to build
     just the module-map-checker tool and its dependencies.

.. _Getting Started with the LLVM System: http://llvm.org/docs/GettingStarted.html
.. _Building LLVM with CMake: http://llvm.org/docs/CMake.html
.. _Clang Tools Documentation: http://clang.llvm.org/docs/ClangTools.html

Module-Map-Checker Usage
========================

``module-map-checker [<module-map-checker-options>] <module-map-file> [<front-end-options>...]``

``<module-map-checker-options>`` is a place-holder for options
specific to module-map-checker, which are described below in
`Module-Map-Checker Command Line Options`.

``<module-map-file>`` specifies the path of a module map
file.  The path can be relative to the current directory.

``<front-end-options>`` is a place-holder for regular Clang
front-end arguments, which must follow the <module-map-file>.

Module-Map-Checker Command Line Options
=======================================

.. option:: -I(include path)

  Look at headers only in this directory tree.
  Must be a path relative to the module.map file.
  There can be multiple ``-I`` options, for when the
  module map covers multiple directories, and
  excludes higher or sibling directories not
  specified. If this option is omitted, the
  directory containing the module-map-file is
  the root of the header tree to be searched for
  headers.

.. option:: -dump-module-map

  Dump the module map object during the check.
  This displays the modules and headers.
