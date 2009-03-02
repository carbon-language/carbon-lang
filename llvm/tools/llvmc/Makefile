##===- tools/llvmc/Makefile --------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..

# The current plan is to make the user copy the skeleton project and change only
# this file (and plugins/UserPlugin, of course).

export LLVMC_BASED_DRIVER_NAME = llvmc
export LLVMC_BUILTIN_PLUGINS = Base Clang

DIRS = plugins driver

include $(LEVEL)/Makefile.common
