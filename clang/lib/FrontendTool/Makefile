##===- clang/lib/FrontendTool/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
LIBRARYNAME := clangFrontendTool

include $(CLANG_LEVEL)/Makefile
include $(CLANG_LEVEL)/../../Makefile.config

ifeq ($(ENABLE_CLANG_ARCMT),1)
  CXX.Flags += -DCLANG_ENABLE_ARCMT
endif

ifeq ($(ENABLE_CLANG_REWRITER),1)
  CXX.Flags += -DCLANG_ENABLE_REWRITER
endif

ifeq ($(ENABLE_CLANG_STATIC_ANALYZER),1)
  CXX.Flags += -DCLANG_ENABLE_STATIC_ANALYZER
endif


