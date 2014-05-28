AddressSanitizer RT
================================
This directory contains sources of the AddressSanitizer (asan) runtime library.
We are in the process of integrating AddressSanitizer with LLVM, stay tuned.

Directory structure:
README.txt       : This file.
Makefile.mk      : File for make-based build.
CMakeLists.txt   : File for cmake-based build.
asan_*.{cc,h}    : Sources of the asan runtime library.
scripts/*        : Helper scripts.
tests/*          : ASan unit tests.

Also ASan runtime needs the following libraries:
lib/interception/      : Machinery used to intercept function calls.
lib/sanitizer_common/  : Code shared between ASan and TSan.

Currently ASan runtime can be built by both make and cmake build systems.
(see compiler-rt/make and files Makefile.mk for make-based build and
files CMakeLists.txt for cmake-based build).

ASan unit and output tests work only with cmake. You may run this
command from the root of your cmake build tree:

make check-asan

For more instructions see:
http://code.google.com/p/address-sanitizer/wiki/HowToBuild
