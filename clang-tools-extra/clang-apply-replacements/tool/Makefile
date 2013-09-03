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

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc mcparser option
USEDLIBS = clangApplyReplacements.a clangFormat.a clangTooling.a clangFrontend.a \
	   clangSerialization.a clangDriver.a clangRewriteFrontend.a \
	   clangRewriteCore.a clangParse.a clangSema.a clangAnalysis.a \
	   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

CPP.Flags += -I$(PROJ_SRC_DIR)/../include
