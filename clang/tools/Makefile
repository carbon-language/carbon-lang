##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..

include $(CLANG_LEVEL)/../../Makefile.config

DIRS := libclang c-index-test arcmt-test c-arcmt-test
PARALLEL_DIRS := driver diagtool clang-format

ifeq ($(ENABLE_CLANG_STATIC_ANALYZER),1)
  PARALLEL_DIRS += clang-check
endif

# Recurse into the extra repository of tools if present.
OPTIONAL_PARALLEL_DIRS := extra

ifeq ($(BUILD_CLANG_ONLY),YES)
  PARALLEL_DIRS := driver libclang c-index-test
  OPTIONAL_DIRS :=
endif

include $(CLANG_LEVEL)/Makefile
