##===- unittests/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# If LLD_LEVEL is not set, then we are the top-level Makefile. Otherwise, we
# are being included from a subdirectory makefile.

ifndef LLD_LEVEL

IS_UNITTEST_LEVEL := 1
LLD_LEVEL := ..

PARALLEL_DIRS = CoreTests DriverTests

include $(LLD_LEVEL)/../../Makefile.config

endif  # LLD_LEVEL

include $(LLD_LEVEL)/Makefile

ifndef IS_UNITTEST_LEVEL

MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest

endif  # IS_UNITTEST_LEVEL
