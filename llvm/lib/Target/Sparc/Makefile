##===- lib/Target/Sparc/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMSparc
TARGET = Sparc
EXTRA_DIST = README.txt

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SparcGenRegisterInfo.h.inc SparcGenRegisterNames.inc \
                SparcGenRegisterInfo.inc SparcGenInstrNames.inc \
                SparcGenInstrInfo.inc SparcGenAsmWriter.inc \
                SparcGenDAGISel.inc SparcGenSubtarget.inc

include $(LEVEL)/Makefile.common

