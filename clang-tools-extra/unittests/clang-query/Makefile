##===- unittests/clang-query/Makefile ----------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TESTNAME = ClangQuery
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader ipo objcarcopts \
                   instrumentation bitwriter support mc option \
		 TransformUtils
USEDLIBS = clangQuery.a clangTooling.a clangFrontend.a clangCodeGen.a clangSerialization.a \
	   clangDriver.a clangParse.a clangSema.a clangEdit.a clangAnalysis.a \
	   clangAST.a clangASTMatchers.a clangDynamicASTMatchers.a clangLex.a \
	   clangBasic.a

include $(CLANG_LEVEL)/Makefile
MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
CPP.Flags += -I$(PROJ_SRC_DIR)/../../clang-query
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
