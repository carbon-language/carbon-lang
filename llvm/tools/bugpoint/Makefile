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
	   LLVMTransformUtils.a \
	   LLVMAsmParser.a LLVMLinker.a LLVMBCReader.a LLVMBCWriter.a \
	   LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
