##===- tools/opt/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = opt
REQUIRES_EH := 1

USEDLIBS = LLVMBCReader.a LLVMBCWriter.a LLVMInstrumentation.a \
	   LLVMScalarOpts.a LLVMipo.a LLVMipa.a LLVMDataStructure LLVMTransforms.a \
	   LLVMTarget.a LLVMTransformUtils.a LLVMAnalysis.a LLVMCore.a LLVMSupport.a \
	   LLVMbzip2.a LLVMSystem.a 

include $(LEVEL)/Makefile.common
