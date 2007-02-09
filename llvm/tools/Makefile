##===- tools/Makefile --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ..
# NOTE: The tools are organized into five groups of four consisting of one 
# large and three small executables. This is done to minimize memory load 
# in parallel builds.  Please retain this ordering.
PARALLEL_DIRS := llvm-config  \
                 opt llvm-as llvm-dis llvm-upgrade \
                 llc llvm-ranlib llvm-ar llvm-nm \
                 llvm-ld llvmc llvm-prof llvm-link \
		 lli gccas gccld llvm-extract llvm-db llvm2cpp \
		 bugpoint llvm-stub llvm-bcanalyzer llvm-stub
		

include $(LEVEL)/Makefile.config

# Disable liblto on Windows until compatability is determined.
ifneq ($(OS), MingW)
PARALLEL_DIRS += lto
endif

include $(LEVEL)/Makefile.common
