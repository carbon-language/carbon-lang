##===- tools/llvm-lto/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := llvm-lto
LINK_COMPONENTS := support

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS := 1

NO_INSTALL := 1

include $(LEVEL)/Makefile.common

LDFLAGS += -L$(LibDir)
LIBS += -lLTO
