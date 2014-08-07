##===- lib/Target/SystemZ/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMSystemZCodeGen
TARGET = SystemZ

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SystemZGenRegisterInfo.inc \
		SystemZGenAsmWriter.inc \
		SystemZGenAsmMatcher.inc \
		SystemZGenDisassemblerTables.inc \
		SystemZGenInstrInfo.inc \
		SystemZGenDAGISel.inc \
		SystemZGenSubtargetInfo.inc \
		SystemZGenCallingConv.inc \
		SystemZGenMCCodeEmitter.inc

DIRS = InstPrinter AsmParser Disassembler TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

