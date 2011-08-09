##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
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

LINK_COMPONENTS := support
 
USEDLIBS = clangCodeGen.a clangParse.a clangSema.a \
           clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

