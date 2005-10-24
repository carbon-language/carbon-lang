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

USEDLIBS = LLVMBCReader LLVMBCWriter LLVMInstrumentation.a LLVMProfilePaths \
	   LLVMScalarOpts.a LLVMipo.a LLVMipa.a LLVMDataStructure LLVMTransforms.a \
	   LLVMTarget.a LLVMAnalysis.a LLVMTransformUtils.a LLVMCore LLVMSupport.a \
	   LLVMbzip2 LLVMSystem.a 

include $(LEVEL)/Makefile.common
