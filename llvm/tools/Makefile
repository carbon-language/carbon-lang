##===- tools/Makefile --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ..
# Note .. the tools are organized in triples consisting of one large and two
# small executables. This is done to minimize memory load in parallel builds.
# Please retain this ordering.
PARALLEL_DIRS := llvm-config \
                 opt llvm-as llvm-dis \
                 llc llvm-ranlib llvm-ar \
		 lli llvm-link llvm-nm \
		 bugpoint llvm-db llvm-extract \
                 gccas llvm-bcanalyzer llvm-stub \
		 gccld llvm2cpp \
                 llvm-ld llvmc llvm-prof

include $(LEVEL)/Makefile.common
