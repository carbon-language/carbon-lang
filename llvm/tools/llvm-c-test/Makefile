##===- tools/llvm-c-test -----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llvm-c-test

TOOL_NO_EXPORTS = 1
NO_INSTALL = 1


# If there is no shared lib, link all components...
ifneq ($(ENABLE_SHARED),1)
LINK_COMPONENTS = all
endif

include $(LEVEL)/Makefile.common

CFLAGS += -std=c99 -Wall -Wstrict-prototypes

# ...but if it is built - use it
ifeq ($(ENABLE_SHARED),1)
LIBS = -lLLVM-$(LLVMVersion)
endif
