##===- lib/Target/PIC16/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source 
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
LIBRARYNAME = PIC16Passes
TARGET = PIC16

LOADABLE_MODULE = 1

# Hack: we need to include 'main' pic16 target directory to grab private headers
CPPFLAGS = -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LEVEL)/Makefile.common

