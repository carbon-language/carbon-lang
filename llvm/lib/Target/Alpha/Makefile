##===- lib/Target/Alpha/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMAlpha
TARGET = Alpha

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AlphaGenRegisterInfo.h.inc AlphaGenRegisterNames.inc \
                AlphaGenRegisterInfo.inc AlphaGenInstrNames.inc \
                AlphaGenInstrInfo.inc AlphaGenCodeEmitter.inc \
                AlphaGenAsmWriter.inc

include $(LEVEL)/Makefile.common
