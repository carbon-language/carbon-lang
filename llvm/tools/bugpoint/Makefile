##===- tools/bugpoint/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = bugpoint

OPTLIBS  = LLVMTransforms LLVMInstrumentation LLVMProfilePaths
ANALIBS  = LLVMDataStructure LLVMipa LLVMTarget.a 

USEDLIBS = LLVMipo LLVMScalarOpts LLVMAnalysis $(OPTLIBS) $(ANALIBS) \
	   LLVMTransformUtils \
	   LLVMAsmParser LLVMLinker.a LLVMBCReader LLVMBCWriter \
	   LLVMCore LLVMSupport.a LLVMbzip2 LLVMSystem.a

include $(LEVEL)/Makefile.common
