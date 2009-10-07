##===- clang/lib/Driver/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME := clangDriver
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti
OS_MAJOR := -DOS_MAJOR=$(shell uname -r | sed 's/\..*//')
# Don't install Clang libraries
NO_INSTALL = 1

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include $(OS_MAJOR)

include $(LEVEL)/Makefile.common
