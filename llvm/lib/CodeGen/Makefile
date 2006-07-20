##===- lib/CodeGen/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMCodeGen
PARALLEL_DIRS = SelectionDAG
BUILD_ARCHIVE = 1

include $(LEVEL)/Makefile.common

CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))
