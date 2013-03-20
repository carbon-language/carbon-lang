##===- clang-format/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..

TOOLNAME = clang-format

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc
USEDLIBS = clangFormat.a clangTooling.a clangFrontend.a clangSerialization.a \
	   clangDriver.a clangParse.a clangSema.a clangAnalysis.a \
           clangRewriteFrontend.a clangRewriteCore.a clangEdit.a clangAST.a \
           clangLex.a clangBasic.a 

include $(CLANG_LEVEL)/Makefile
