##===- lib/Target/Sparc/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMSparc
TARGET = Sparc

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SparcGenRegisterInfo.h.inc SparcGenRegisterNames.inc \
                SparcGenRegisterInfo.inc SparcGenInstrNames.inc \
                SparcGenInstrInfo.inc SparcGenAsmWriter.inc \
                SparcGenDAGISel.inc SparcGenSubtarget.inc SparcGenCallingConv.inc

include $(LEVEL)/Makefile.common

