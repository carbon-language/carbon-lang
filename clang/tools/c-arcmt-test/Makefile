##===- tools/c-arcmt-test/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = c-arcmt-test

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Don't install this. It is used for tests.
NO_INSTALL = 1

LINK_COMPONENTS := support mc
USEDLIBS = clang.a clangIndex.a clangFrontend.a clangDriver.a \
	   clangSerialization.a clangParse.a clangSema.a \
	   clangAnalysis.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile
