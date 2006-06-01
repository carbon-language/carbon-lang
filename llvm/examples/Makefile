##===- examples/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL=..

include $(LEVEL)/Makefile.config

PARALLEL_DIRS:= Fibonacci HowToUseJIT ModuleMaker BFtoLLVM

ifeq ($(ENABLE_THREADS),1)
PARALLEL_DIRS += ParallelJIT 
endif

include $(LEVEL)/Makefile.common
