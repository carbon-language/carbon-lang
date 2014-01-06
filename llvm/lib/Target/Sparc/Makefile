##===- lib/Target/Sparc/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMSparcCodeGen
TARGET = Sparc

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SparcGenRegisterInfo.inc SparcGenInstrInfo.inc \
		SparcGenAsmWriter.inc SparcGenAsmMatcher.inc \
		SparcGenDAGISel.inc SparcGenDisassemblerTables.inc \
		SparcGenSubtargetInfo.inc SparcGenCallingConv.inc \
		SparcGenCodeEmitter.inc SparcGenMCCodeEmitter.inc

DIRS = InstPrinter AsmParser Disassembler TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

