##===- tools/diagtool/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = diagtool 

# No plugins, optimize startup time.
TOOL_NO_EXPORTS := 1

# Don't install this.
NO_INSTALL = 1

LINK_COMPONENTS := support mc

USEDLIBS = clangAnalysis.a clangAST.a clangBasic.a clangDriver.a clangEdit.a \
           clangFrontend.a clangLex.a clangParse.a clangSema.a \
           clangSerialization.a

include $(CLANG_LEVEL)/Makefile

