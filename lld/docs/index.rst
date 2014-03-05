.. _index:

lld - The LLVM Linker
=====================

lld is a new set of modular code for creating linker tools.

* End-User Features:

  * Compatible with existing linker options
  * Reads standard Object Files (e.g. ELF, Mach-O, PE/COFF)
  * Writes standard Executable Files (e.g. ELF, Mach-O, PE)
  * Fast link times
  * Minimal memory use
  * Remove clang's reliance on "the system linker"
  * Uses the LLVM `"UIUC" BSD-Style license`__.

* Applications:

  * Modular design
  * Support cross linking
  * Easy to add new CPU support
  * Can be built as static tool or library

* Design and Implementation:

  * Extensive unit tests
  * Internal linker model can be dumped/read to textual format
  * Internal linker model can be dumped/read to a new native format
  * Native format designed to be fast to read and write
  * Additional linking features can be plugged in as "passes"
  * OS specific and CPU specific code factored out

Why a new linker?
-----------------

The fact that clang relies on whatever linker tool you happen to have installed
means that clang has been very conservative adopting features which require a
recent linker.

In the same way that the MC layer of LLVM has removed clang's reliance on the
system assembler tool, the lld project will remove clang's reliance on the
system linker tool.


Current Status
--------------

lld is in its early stages of development.

It can currently self host on Linux x86-64 with -static.

Source
------

lld is available in the LLVM SVN repository::

  svn co http://llvm.org/svn/llvm-project/lld/trunk

lld is also available via the read-only git mirror::

  git clone http://llvm.org/git/lld.git

Contents
--------

.. toctree::
   :maxdepth: 2

   design
   getting_started
   development
   windows_support
   open_projects
   sphinx_intro

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

__ http://llvm.org/docs/DeveloperPolicy.html#license
