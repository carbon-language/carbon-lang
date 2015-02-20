##===- clang-apply-replacements/tool/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TOOLNAME = clang-apply-replacements

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

SOURCES = ClangApplyReplacementsMain.cpp

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter \
                   instrumentation ipo support mc objcarcopts option
USEDLIBS = clangApplyReplacements.a clangFormat.a \
	   clangTooling.a clangToolingCore.a clangFrontend.a \
	   clangSerialization.a clangDriver.a clangRewriteFrontend.a \
	   clangRewrite.a clangParse.a clangSema.a clangAnalysis.a \
	   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a \
           clangCodeGen.a

include $(CLANG_LEVEL)/Makefile

CPP.Flags += -I$(PROJ_SRC_DIR)/../include
