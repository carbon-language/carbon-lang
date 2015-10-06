The New ELF Linker
==================
This directory contains a port of the new PE/COFF linker for ELF.

Overall Design
--------------
See COFF/README.md for details on the design. Note that unlike COFF, we do not
distinguish chunks from input sections; they are merged together.

Capabilities
------------
This linker can link LLVM and Clang on Linux x86-64 with -LLVM_ENABLE_THREADS=OFF.
