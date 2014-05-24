##===- lib/Target/AArch64/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMAArch64CodeGen
TARGET = AArch64

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AArch64GenRegisterInfo.inc AArch64GenInstrInfo.inc \
		AArch64GenAsmWriter.inc AArch64GenAsmWriter1.inc \
		AArch64GenDAGISel.inc \
		AArch64GenCallingConv.inc AArch64GenAsmMatcher.inc \
		AArch64GenSubtargetInfo.inc AArch64GenMCCodeEmitter.inc \
		AArch64GenFastISel.inc AArch64GenDisassemblerTables.inc \
		AArch64GenMCPseudoLowering.inc

DIRS = TargetInfo InstPrinter AsmParser Disassembler MCTargetDesc Utils

include $(LEVEL)/Makefile.common
