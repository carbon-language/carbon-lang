##===- unittests/CodeGen/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = CodeGen
include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter \
                   instrumentation ipo mc option objcarcopts profiledata support
USEDLIBS = clangFrontend.a clangCodeGen.a clangSerialization.a \
           clangDriver.a \
           clangParse.a clangSema.a clangAnalysis.a \
           clangEdit.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
