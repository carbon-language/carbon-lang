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
LINK_COMPONENTS := engine interpreter

include $(LEVEL)/Makefile.config

SOURCES := ExecutionEngineTest.cpp \
  JITEventListenerTest.cpp

ifeq ($(USE_INTEL_JITEVENTS), 1)
  # Build the Intel JIT Events interface tests
  SOURCES += IntelJITEventListenerTest.cpp

  # Add the Intel JIT Events include directory
  CPPFLAGS += -I$(INTEL_JITEVENTS_INCDIR)

  # Link against the LLVM Intel JIT Evens interface library
  LINK_COMPONENTS += inteljitevents
endif

ifeq ($(USE_OPROFILE), 1)
  # Build the OProfile JIT interface tests
  SOURCES += OProfileJITEventListenerTest.cpp

  # Link against the LLVM oprofile interface library
  LINK_COMPONENTS += oprofilejit
endif


PARALLEL_DIRS = JIT

include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
