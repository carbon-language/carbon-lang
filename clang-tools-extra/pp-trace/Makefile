##===- extra/pp-trace/Makefile --------------------------*- Makefile -*---===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===---------------------------------------------------------------------===##

CLANG_LEVEL := ../../..

TOOLNAME = pp-trace

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser mcparser bitreader bitwriter \
                   instrumentation ipo support transformutils mc objcarcopts option

USEDLIBS = clangFrontend.a clangSerialization.a clangDriver.a \
           clangTooling.a clangParse.a clangSema.a clangAnalysis.a \
           clangEdit.a clangAST.a clangLex.a clangBasic.a clangCodeGen.a

include $(CLANG_LEVEL)/Makefile
