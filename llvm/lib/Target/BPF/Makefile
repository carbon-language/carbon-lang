##===- lib/Target/BPF/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMBPFCodeGen
TARGET = BPF

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = BPFGenRegisterInfo.inc BPFGenInstrInfo.inc \
		BPFGenAsmWriter.inc BPFGenAsmMatcher.inc BPFGenDAGISel.inc \
		BPFGenMCCodeEmitter.inc BPFGenSubtargetInfo.inc BPFGenCallingConv.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
