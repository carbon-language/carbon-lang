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
USEDLIBS = LLVMArchive.a LLVMBCReader.a \
	   LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

check-local::
	$(Echo) Checking llvm-ar
	$(Verb) $(ToolDir)/llvm-ar zRrS nada.a .
	$(Verb) $(ToolDir)/llvm-ar tv nada.a | \
	  grep Debug/llvm-ar.d >/dev/null 2>&1
	$(Verb) $(RM) -f nada.a
