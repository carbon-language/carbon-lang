##===- lib/Target/PowerPC/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMPowerPC
TARGET = PowerPC PPC32

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = PowerPCGenInstrNames.inc PowerPCGenRegisterNames.inc \
                PowerPCGenAsmWriter.inc  PPC32GenCodeEmitter.inc \
                PPC32GenRegisterInfo.h.inc PPC32GenRegisterInfo.inc \
                PPC32GenInstrInfo.inc PPC32GenDAGISel.inc

include $(LEVEL)/Makefile.common
