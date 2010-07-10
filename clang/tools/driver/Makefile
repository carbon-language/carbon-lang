##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = clang
ifndef CLANG_IS_PRODUCTION
TOOLALIAS = clang++
else
  ifdef CLANGXX_IS_PRODUCTION
    TOOLALIAS = clang++
  endif
endif

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter codegen \
                   ipo selectiondag
USEDLIBS = clangFrontend.a clangDriver.a clangCodeGen.a clangSema.a \
           clangChecker.a clangAnalysis.a clangRewrite.a  clangAST.a \
           clangParse.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

# Translate make variable to define when building a "production" clang.
ifdef CLANG_IS_PRODUCTION
CPP.Defines += -DCLANG_IS_PRODUCTION
endif
ifdef CLANGXX_IS_PRODUCTION
CPP.Defines += -DCLANGXX_IS_PRODUCTION
endif
