##===- lib/Target/ARM64/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMARM64CodeGen
TARGET = ARM64

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = ARM64GenRegisterInfo.inc ARM64GenInstrInfo.inc \
		ARM64GenAsmWriter.inc ARM64GenAsmWriter1.inc \
		ARM64GenDAGISel.inc \
		ARM64GenCallingConv.inc ARM64GenAsmMatcher.inc \
		ARM64GenSubtargetInfo.inc ARM64GenMCCodeEmitter.inc \
		ARM64GenFastISel.inc ARM64GenDisassemblerTables.inc \
		ARM64GenMCPseudoLowering.inc

DIRS = TargetInfo InstPrinter AsmParser Disassembler MCTargetDesc Utils

include $(LEVEL)/Makefile.common
