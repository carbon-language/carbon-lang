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

ifeq ($(TARGET_HAS_JIT),1)
	PARALLEL_DIRS = JIT MCJIT
endif

include $(LEVEL)/Makefile.config
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
