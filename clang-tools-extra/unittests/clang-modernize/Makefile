##===- unittests/clang-modernize/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TESTNAME = ClangModernizeTests
LINK_COMPONENTS := asmparser bitreader support MC MCParser option \
		TransformUtils
USEDLIBS = modernizeCore.a clangFormat.a clangApplyReplacements.a \
	   clangTooling.a clangToolingCore.a clangFrontend.a \
           clangSerialization.a clangDriver.a clangRewriteFrontend.a \
           clangRewrite.a clangParse.a clangSema.a clangAnalysis.a \
           clangAST.a clangASTMatchers.a clangEdit.a clangLex.a \
           clangBasic.a

include $(CLANG_LEVEL)/Makefile
MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
CPP.Flags += -I$(PROJ_SRC_DIR)/../../clang-modernize -I$(PROJ_SRC_DIR)/../../clang-apply-replacements/include \
	           -I$(PROJ_SRC_DIR)/../include
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
