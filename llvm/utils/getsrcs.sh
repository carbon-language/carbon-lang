#!/bin/sh
# This is useful because it prints out all of the source files.  Useful for
# greps.
find docs include lib tools utils projects -name \*.\[cdhyl\]\* | grep -v Lexer.cpp | \
       grep -v llvmAsmParser.cpp | grep -v llvmAsmParser.h | grep -v '~$' | \
       grep -v '\.ll$' | grep -v .flc | grep -v Sparc.burm.c | grep -v '\.d$' |\
       grep -v '\.dir$' | grep -v www/docs/doxygen | grep -v include/boost | \
       grep -v /Burg/ | grep -v /Spiff/ | grep -v '\.lo' | grep -v '\.inc$' | grep -v '\.libs' | \
       grep -v TableGen/FileParser.cpp | grep -v TableGen/FileParser.h

