#!/bin/sh
# This is useful because it prints out all of the source files.  Useful for
# greps.
find docs include lib tools -name \*.\[cdhyl\]\* | grep -v Lexer.cpp | grep -v llvmAsmParser.cpp | grep -v llvmAsmParser.h | grep -v '~$' | grep -v '\.ll$' | grep -v .flc | grep -v Sparc.burm.c | grep -v '\.d$' | grep -v '\.dir$' | grep -v docs/doxygen | grep -v include/boost

