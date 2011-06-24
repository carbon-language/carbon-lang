##===- lib/Target/CellSPU/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMCellSPUCodeGen
TARGET = SPU
BUILT_SOURCES = SPUGenInstrNames.inc \
		SPUGenAsmWriter.inc SPUGenCodeEmitter.inc \
		SPUGenRegisterNames.inc SPUGenRegisterDesc.inc \
		SPUGenRegisterInfo.h.inc SPUGenRegisterInfo.inc \
		SPUGenInstrInfo.inc SPUGenDAGISel.inc \
		SPUGenSubtarget.inc SPUGenCallingConv.inc

DIRS = TargetInfo

include $(LEVEL)/Makefile.common
