##===- clang/lib/CodeGen/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This implements the AST -> LLVM code generation library for the 
# C-Language front-end.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME := clangCodeGen
BUILD_ARCHIVE = 1

CPP.Flags += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
ifdef CLANG_VENDOR
CPP.Flags += -DCLANG_VENDOR='"$(CLANG_VENDOR) "'
endif

include $(LEVEL)/Makefile.common

