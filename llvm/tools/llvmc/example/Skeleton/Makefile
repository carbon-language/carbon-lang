##===- llvmc/example/Skeleton/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# Change this so that $(BASE_LEVEL)/Makefile.common refers to
# $LLVM_DIR/Makefile.common.
export LLVMC_BASE_LEVEL = ../../../..

# Change this to the name of your LLVMC-based driver.
export LLVMC_BASED_DRIVER_NAME = llvmc-skeleton

# List your plugin names here
export LLVMC_BUILTIN_PLUGINS = # Plugin

LEVEL = $(LLVMC_BASE_LEVEL)

DIRS = plugins driver

include $(LEVEL)/Makefile.common
