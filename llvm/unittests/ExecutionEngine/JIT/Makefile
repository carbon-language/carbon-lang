##===- unittests/ExecutionEngine/JIT/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
TESTNAME = JIT
LINK_COMPONENTS := asmparser bitreader bitwriter core jit native support

include $(LEVEL)/Makefile.config
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest

# Permit these tests to use the JIT's symbolic lookup.
LD.Flags += $(RDYNAMIC)
