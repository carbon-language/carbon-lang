#!/bin/sh
# This is useful because it prints out all of the source files.  Useful for
# greps.
find . -name \*.\[chyl\]\* | grep -v Lexer.cpp | grep -v llvmAsmParser.cpp | grep -v llvmAsmParser.h | grep -v '~$' | grep -v '\.ll$' | grep -v test | grep -v .flc
