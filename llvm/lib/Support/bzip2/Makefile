##===- lib/Support/bzip2/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMbzip2
SOURCES= blocksort.c bzlib.c
SOURCES=blocksort.c huffman.c crctable.c randtable.c compress.c decompress.c \
	bzlib.c
CFLAGS=-fno-strength-reduce 

include $(LEVEL)/Makefile.common
