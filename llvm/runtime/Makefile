##===- runtime/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
# llvm/runtime Makefile:  Build all subdirectories automatically

LEVEL = ..
PARALLEL_DIRS  := $(sort $(filter-out Output/, $(filter-out CVS/, $(wildcard */))))

include $(LEVEL)/Makefile.common

# Install target for libraries: Copy into the gcc install directory in chris's
# tree...
#
install::

clean::
