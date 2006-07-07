##===- tools/llvm-ld/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

TOOLNAME = llvm-ld
USEDLIBS = LLVMipo.a LLVMTransforms.a LLVMScalarOpts.a LLVMAnalysis.a \
	   LLVMipa.a LLVMTransformUtils.a LLVMTarget.a LLVMLinker.a \
	   LLVMArchive.a LLVMBCReader.a LLVMBCWriter.a \
	   LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
