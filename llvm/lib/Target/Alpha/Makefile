##===- lib/Target/Alpha/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMAlphaCodeGen
TARGET = Alpha

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AlphaGenRegisterInfo.inc AlphaGenInstrInfo.inc \
                AlphaGenAsmWriter.inc AlphaGenDAGISel.inc \
                AlphaGenCallingConv.inc AlphaGenSubtarget.inc

DIRS = TargetInfo

include $(LEVEL)/Makefile.common
