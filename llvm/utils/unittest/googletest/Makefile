##===- utils/unittest/googletest/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../../..

include $(LEVEL)/Makefile.config

LIBRARYNAME = GoogleTest
BUILD_ARCHIVE = 1
REQUIRES_RTTI = 1
CPP.Flags += -I$(LLVM_SRC_ROOT)/utils/unittest/googletest/include
CPP.Flags += $(NO_MISSING_FIELD_INITIALIZERS) $(NO_VARIADIC_MACROS)

ifeq ($(HOST_OS),MingW)
  CPP.Flags += -DGTEST_OS_WINDOWS=1
endif

NO_INSTALL = 1

include $(LEVEL)/Makefile.common
