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

OPTLIBS  = LLVMTransforms.a LLVMInstrumentation.a
ANALIBS  = LLVMDataStructure LLVMipa.a LLVMTarget.a 

USEDLIBS = LLVMipo.a LLVMScalarOpts.a $(OPTLIBS) $(ANALIBS) LLVMAnalysis.a \
	   LLVMTransformUtils \
	   LLVMAsmParser LLVMLinker.a LLVMBCReader LLVMBCWriter \
	   LLVMCore LLVMSupport.a LLVMbzip2 LLVMSystem.a

include $(LEVEL)/Makefile.common
