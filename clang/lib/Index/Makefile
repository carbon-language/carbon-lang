##===- clang/lib/Index/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the Indexer library for the C-Language front-end.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
include $(CLANG_LEVEL)/../../Makefile.config

LIBRARYNAME := clangIndex
BUILD_ARCHIVE = 1

ifeq ($(ARCH),PowerPC)
CXX.Flags += -maltivec
endif

CPP.Flags += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include

include $(CLANG_LEVEL)/Makefile

