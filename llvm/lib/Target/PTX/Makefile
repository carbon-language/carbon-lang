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
		PTXGenRegisterInfo.inc \
		PTXGenSubtargetInfo.inc

DIRS = TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
