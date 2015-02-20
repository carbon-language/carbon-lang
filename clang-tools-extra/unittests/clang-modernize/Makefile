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
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser mcparser bitreader bitwriter \
                   instrumentation ipo support transformutils mc objcarcopts option
USEDLIBS = modernizeCore.a clangFormat.a clangApplyReplacements.a \
	   clangTooling.a clangToolingCore.a clangFrontend.a \
           clangSerialization.a clangDriver.a clangRewriteFrontend.a \
           clangRewrite.a clangParse.a clangSema.a clangAnalysis.a \
           clangAST.a clangASTMatchers.a clangEdit.a clangLex.a \
           clangBasic.a clangCodeGen.a

include $(CLANG_LEVEL)/Makefile
MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
CPP.Flags += -I$(PROJ_SRC_DIR)/../../clang-modernize -I$(PROJ_SRC_DIR)/../../clang-apply-replacements/include \
	           -I$(PROJ_SRC_DIR)/../include
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
