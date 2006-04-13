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
TARGET = PPC
EXTRA_DIST = README.txt README_ALTIVEC.txt

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = PPCGenInstrNames.inc PPCGenRegisterNames.inc \
                PPCGenAsmWriter.inc  PPCGenCodeEmitter.inc \
                PPCGenRegisterInfo.h.inc PPCGenRegisterInfo.inc \
                PPCGenInstrInfo.inc PPCGenDAGISel.inc PPCGenSubtarget.inc

include $(LEVEL)/Makefile.common
