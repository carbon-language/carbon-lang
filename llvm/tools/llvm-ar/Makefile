##===- tools/llvm-ar/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = llvm-ar 
USEDLIBS = LLVMArchive.a LLVMBCReader \
	   LLVMCore LLVMSupport.a LLVMbzip2 LLVMSystem.a

include $(LEVEL)/Makefile.common

check-local::
	$(ToolDir)/llvm-ar zRrS nada.a .
	$(ToolDir)/llvm-ar tv nada.a | grep Debug/llvm-ar.d >/dev/null 2>&1
	$(RM) -f nada.a
