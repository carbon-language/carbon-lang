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
LINK_COMPONENTS := support mc
USEDLIBS = clangTooling.a clangFrontend.a clangSerialization.a clangDriver.a \
           clangParse.a clangRewrite.a clangSema.a clangAnalysis.a clangEdit.a \
           clangAST.a clangASTMatchers.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
