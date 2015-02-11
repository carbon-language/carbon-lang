##===- source/Host/windows/Makefile ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../..
LIBRARYNAME := lldbHostWindows
BUILD_ARCHIVE = 1

SOURCES = $(notdir $(wildcard $(PROJ_SRC_DIR)/*.cpp )) $(PROJ_SRC_DIR)/../posix/ConnectionFileDescriptorPosix.cpp

include $(LLDB_LEVEL)/Makefile
