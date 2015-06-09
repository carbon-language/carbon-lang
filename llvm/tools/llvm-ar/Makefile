##===- tools/llvm-ar/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := llvm-ar
TOOLALIAS = llvm-ranlib
LINK_COMPONENTS := all-targets bitreader libdriver support object

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS := 1

include $(LEVEL)/Makefile.common

all-local::
	$(Verb) $(AliasTool) $(notdir $(ToolBuildPath)) $(ToolDir)/llvm-lib$(EXEEXT)
