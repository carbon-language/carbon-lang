##===- lib/Target/SparcV8/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMSparcV8
TARGET = SparcV8

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SparcV8GenRegisterInfo.h.inc SparcV8GenRegisterNames.inc \
                SparcV8GenRegisterInfo.inc SparcV8GenInstrNames.inc \
                SparcV8GenInstrInfo.inc SparcV8GenAsmWriter.inc \
                SparcV8GenDAGISel.inc SparcV8GenSubtarget.inc

include $(LEVEL)/Makefile.common

