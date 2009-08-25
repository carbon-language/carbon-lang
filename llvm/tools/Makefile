##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ..

# Build clang if present.
OPTIONAL_PARALLEL_DIRS := clang

# NOTE: The tools are organized into five groups of four consisting of one
# large and three small executables. This is done to minimize memory load
# in parallel builds.  Please retain this ordering.
DIRS := llvm-config
PARALLEL_DIRS := opt llvm-as llvm-dis \
                 llc llvm-ranlib llvm-ar llvm-nm \
                 llvm-ld llvm-prof llvm-link \
                 lli llvm-extract llvm-db \
                 bugpoint llvm-bcanalyzer llvm-stub \
                 llvm-mc llvmc

# Let users override the set of tools to build from the command line.
ifdef ONLY_TOOLS
  OPTIONAL_PARALLEL_DIRS :=
  PARALLEL_DIRS := $(ONLY_TOOLS)
endif

include $(LEVEL)/Makefile.config

ifeq ($(ENABLE_PIC),1)
  DIRS += lto
  ifdef BINUTILS_INCDIR
    DIRS += gold
  endif
endif

# No support for lto / gold on windows targets
ifeq ($(TARGET_OS), $(filter $(TARGET_OS), Cygwin MingW))
  DIRS := $(filter-out lto gold, $(DIRS))
endif

include $(LEVEL)/Makefile.common
