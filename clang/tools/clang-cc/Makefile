##===- tools/clang-cc/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..

TOOLNAME = clang-cc
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti

# Clang has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) bitreader bitwriter codegen ipo selectiondag
USEDLIBS = clangDriver.a clangFrontend.a clangCodeGen.a clangAnalysis.a	\
           clangRewrite.a clangSema.a clangAST.a clangParse.a 		\
           clangLex.a clangBasic.a 

# clang-cc lives in a special location; we can get away with this
# because nothing else gets installed from here.
PROJ_bindir := $(DESTDIR)$(PROJ_prefix)/libexec

include $(LLVM_SRC_ROOT)/Makefile.rules
