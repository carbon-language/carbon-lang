##===- tools/analyze/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = analyze
USEDLIBS = LLVMAsmParser.a LLVMBCReader.a LLVMAnalysis.a LLVMipa.a \
           LLVMDataStructure \
	   LLVMScalarOpts.a LLVMTransforms.a LLVMTarget.a LLVMScalarOpts.a \
	   LLVMTransformUtils.a LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
