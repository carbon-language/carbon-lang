##===- tools/debugserver/source/MacOSX/i386/Makefile -------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../../../..

LIBRARYNAME := lldbDebugserverMacOSX_X86_64
BUILD_ARCHIVE = 1

SOURCES := DNBArchImplX86_64.cpp

include $(LLDB_LEVEL)/Makefile

CPP.Flags += -I$(PROJ_SRC_DIR)/.. -I$(PROJ_SRC_DIR)/../.. -I$(PROJ_OBJ_DIR)/../../..