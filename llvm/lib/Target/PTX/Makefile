##===- lib/Target/PTX/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMPTXCodeGen
TARGET = PTX

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = PTXGenAsmWriter.inc \
		PTXGenCallingConv.inc \
		PTXGenDAGISel.inc \
		PTXGenInstrInfo.inc \
		PTXGenInstrNames.inc \
		PTXGenRegisterInfo.inc \
		PTXGenSubtarget.inc

DIRS = TargetInfo

include $(LEVEL)/Makefile.common
