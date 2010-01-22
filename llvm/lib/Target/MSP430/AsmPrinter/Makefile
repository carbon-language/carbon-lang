##===- lib/Target/MSP430/AsmPrinter/Makefile ---------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
LIBRARYNAME = LLVMMSP430AsmPrinter
CXXFLAGS = -fno-rtti

# Hack: we need to include 'main' MSP430 target directory to grab private headers
CPPFLAGS = -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LEVEL)/Makefile.common
