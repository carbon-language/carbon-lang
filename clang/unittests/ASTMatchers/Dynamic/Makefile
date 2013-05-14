##===- unittests/ASTMatchers/Dynamic/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../../..

TESTNAME = DynamicASTMatchers
#include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := support mc
USEDLIBS = clangEdit.a clangTooling.a clangFrontend.a clangSerialization.a \
           clangDriver.a clangRewrite.a clangParse.a clangSema.a \
           clangAnalysis.a clangAST.a clangASTMatchers.a clangLex.a \
           clangBasic.a clangDynamicASTMatchers.a

include $(CLANG_LEVEL)/unittests/Makefile
