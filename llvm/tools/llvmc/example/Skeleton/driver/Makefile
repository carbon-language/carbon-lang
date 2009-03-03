##===- llvmc/example/Skeleton/driver/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = $(LLVMC_BASE_LEVEL)/..

TOOLNAME = $(LLVMC_BASED_DRIVER_NAME)
LLVMLIBS = CompilerDriver

ifneq ($(LLVMC_BUILTIN_PLUGINS),)
USEDLIBS += $(patsubst %,plugin_llvmc_%,$(LLVMC_BUILTIN_PLUGINS))
endif

LINK_COMPONENTS = support system
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
