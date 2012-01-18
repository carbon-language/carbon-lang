AddressSanitizer RT
================================
This directory contains sources of the AddressSanitizer (asan) run-time library.
We are in the process of integrating AddressSanitizer with LLVM, stay tuned.

Directory structre:

README.txt       : This file.
Makefile.mk      : Currently a stub for a proper makefile. not usable.
Makefile.old     : Old out-of-tree makefile, the only usable one so far.
asan_*.{cc,h}    : Sources of the asan run-time lirbary.
mach_override/*  : Utility to override functions on Darwin (MIT License).
scripts/*        : Helper scripts.

Temporary build instructions (verified on linux):

cd lib/asan
make -f Makefile.old get_third_party  # gets googletest and cpplint
make -f Makefile.old test -j 8 CLANG_BUILD=/path/to/Release+Asserts
# Optional:
# make -f Makefile.old install # installs clang and rt to lib/asan_clang_linux

For more info see http://code.google.com/p/address-sanitizer/


