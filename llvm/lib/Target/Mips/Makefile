##===- lib/Target/Mips/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMMipsCodeGen
TARGET = Mips

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MipsGenRegisterInfo.h.inc MipsGenRegisterNames.inc \
                MipsGenRegisterInfo.inc MipsGenRegisterDesc.inc \
		MipsGenInstrNames.inc \
                MipsGenInstrInfo.inc MipsGenAsmWriter.inc \
                MipsGenDAGISel.inc MipsGenCallingConv.inc \
                MipsGenSubtarget.inc

DIRS = TargetInfo

include $(LEVEL)/Makefile.common

