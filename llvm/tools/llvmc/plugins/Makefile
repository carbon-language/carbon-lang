##===- tools/llvmc/plugins/Makefile ------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..

ifneq ($(LLVMC_BUILTIN_PLUGINS),)
DIRS = $(LLVMC_BUILTIN_PLUGINS)
endif

export LLVMC_BUILTIN_PLUGIN=1

include $(LEVEL)/Makefile.common
