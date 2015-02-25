##===- unittests/Tooling/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = Tooling
include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter \
                   instrumentation ipo mc option objcarcopts support
USEDLIBS = clangTooling.a clangToolingCore.a clangFrontend.a \
	   clangCodeGen.a clangSerialization.a clangDriver.a \
           clangParse.a clangRewrite.a clangRewriteFrontend.a \
	   clangSema.a clangAnalysis.a clangEdit.a \
           clangAST.a clangASTMatchers.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
