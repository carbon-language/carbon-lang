##===-------------- linker-script-test/Makefile ----------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===--------------------------------------------------------------------===##

LLD_LEVEL := ../..

TOOLNAME = linker-script-test

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

#include /Makefile.config
LEVEL := $(LLD_LEVEL)/../..
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD)
USEDLIBS = lldReaderWriter.a lldCore.a LLVMSupport.a

include $(LLD_LEVEL)/Makefile
