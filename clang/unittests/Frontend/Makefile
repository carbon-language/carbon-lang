##===- unittests/Frontend/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = Frontend
LINK_COMPONENTS := support mc
USEDLIBS = clangFrontendTool.a clangFrontend.a clangDriver.a \
           clangSerialization.a clangCodeGen.a clangParse.a clangSema.a \
           clangStaticAnalyzerCheckers.a clangStaticAnalyzerCore.a \
           clangARCMigrate.a clangRewrite.a clangEdit.a \
           clangAnalysis.a clangAST.a clangLex.a clangComments.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
