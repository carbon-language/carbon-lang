##===- lib/Target/X86/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMX86
TARGET = X86
EXTRA_DIST = README.txt
# Make sure that tblgen is run, first thing.
BUILT_SOURCES = X86GenRegisterInfo.h.inc X86GenRegisterNames.inc \
                X86GenRegisterInfo.inc X86GenInstrNames.inc \
                X86GenInstrInfo.inc X86GenAsmWriter.inc \
                X86GenAsmWriter1.inc X86GenDAGISel.inc  \
                X86GenSubtarget.inc

include $(LEVEL)/Makefile.common
