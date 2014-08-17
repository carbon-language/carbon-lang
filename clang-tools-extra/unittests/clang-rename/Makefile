##===- unittests/clang-rename/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TESTNAME = ClangRenameTests
LINK_COMPONENTS := asmparser bitreader support MC MCParser option \
		 TransformUtils
USEDLIBS = clangAnalysis.a clangAST.a clangBasic.a clangDriver.a clangEdit.a \
	   clangFrontend.a clangFrontendTool.a clangIndex.a clangLex.a \
	   clangParse.a clangRewrite.a clangRewriteFrontend.a \
	   clangSerialization.a clangSema.a clangTooling.a

include $(CLANG_LEVEL)/Makefile
MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
CPP.Flags += -I(PROJ_SRC_DIR)/../../clang-rename
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
