##===- unittests/ExecutionEngine/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
TESTNAME = ExecutionEngine
LINK_COMPONENTS :=interpreter

include $(LEVEL)/Makefile.config

ifeq ($(TARGET_HAS_JIT),1)
  PARALLEL_DIRS = MCJIT
endif

include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
