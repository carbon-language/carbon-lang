##===- runtime/libdummy/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
BYTECODE_LIBRARY=1
DONT_BUILD_RELINKED=1
LIBRARYNAME=png

Source = png.c pngset.c pngget.c pngrutil.c pngtrans.c pngwutil.c \
	       pngread.c pngrio.c pngwio.c pngwrite.c pngrtran.c \
	       pngwtran.c pngmem.c pngerror.c pngpread.c

include $(LEVEL)/Makefile.common
