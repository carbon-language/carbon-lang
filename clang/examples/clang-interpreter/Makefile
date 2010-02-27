##===- examples/clang-interpreter/Makefile -----------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..

TOOLNAME = clang-interpreter
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
NO_INSTALL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := jit interpreter nativecodegen bitreader bitwriter ipo \
	selectiondag
USEDLIBS = clangFrontend.a clangDriver.a clangCodeGen.a clangSema.a \
           clangChecker.a clangAnalysis.a clangRewrite.a  clangAST.a \
           clangParse.a clangLex.a clangBasic.a

include $(LLVM_SRC_ROOT)/Makefile.rules
