##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../../..

TOOLNAME = clang
TOOLALIAS = clang++
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti

# Clang tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) bitreader bitwriter codegen ipo selectiondag
USEDLIBS = clangFrontend.a clangDriver.a clangCodeGen.a clangAnalysis.a	\
           clangRewrite.a clangSema.a clangAST.a clangParse.a 		\
           clangLex.a clangBasic.a

include $(LLVM_SRC_ROOT)/Makefile.rules

# Translate make variable to define when building a "production" clang.
ifdef CLANG_IS_PRODUCTION
CPP.Defines += -DCLANG_IS_PRODUCTION
endif

