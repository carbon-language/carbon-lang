##===- lib/Target/PTX/AsmPrinter/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
LIBRARYNAME = LLVMPTXAsmPrinter

# Hack: we need to include 'main' PTX target directory to grab private headers
CPP.Flags += -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LEVEL)/Makefile.common
