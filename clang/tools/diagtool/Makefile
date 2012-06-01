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

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := support $(TARGETS_TO_BUILD)

# FIXME: diagtool should be much lighter than this.
USEDLIBS = clangAST.a clangBasic.a clangDriver.a clangEdit.a clangFrontend.a \
           clangLex.a clangParse.a clangSema.a clangSerialization.a

include $(CLANG_LEVEL)/Makefile

