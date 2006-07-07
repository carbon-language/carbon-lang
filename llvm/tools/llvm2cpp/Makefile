##===- tools/llvm2cpp/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = llvm2cpp
USEDLIBS = LLVMBCReader.a LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))
