##===- lld/lib/ReaderWriter/ELF/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLD_LEVEL := ../../..
LIBRARYNAME := lldELF
USEDLIBS = lldPasses.a

CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLD_LEVEL)/lib/ReaderWriter/ELF

# these link against this lib
PARALLEL_DIRS := Hexagon PPC X86 X86_64 Mips AArch64 ARM

include $(LLD_LEVEL)/Makefile
