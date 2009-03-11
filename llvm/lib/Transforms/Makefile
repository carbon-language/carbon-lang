##===- lib/Transforms/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
PARALLEL_DIRS = Utils Instrumentation Scalar IPO Hello

# No support for plugins on windows targets
ifeq ($(OS), $(filter $(OS), Cygwin MingW))
  PARALLEL_DIRS := $(filter-out Hello, $(DIRS))
endif

include $(LEVEL)/Makefile.common

