##===- tools/llvm-as/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = llvm2cpp
USEDLIBS = LLVMAsmParser LLVMBCWriter LLVMCore \
           LLVMSupport.a LLVMbzip2 LLVMSystem.a

include $(LEVEL)/Makefile.common

tryit: all-local recurty.cpp globalvars.cpp

%.cpp : %.ll
	llvm2cpp $*.ll -f -o $*.cpp
	gcc -I$(LLVM_SRC_ROOT)/include -I$(LLVM_OBJ_ROOT)/include -g \
	-D__STDC_LIMIT_MACROS -L$(LibDir) $(LibDir)/LLVMCore.o -lLLVMSupport \
	$(LibDir)/LLVMbzip2.o -lLLVMSystem -lstdc++ \
	$*.cpp -o $*
