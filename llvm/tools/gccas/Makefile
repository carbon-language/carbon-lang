##===- tools/gccas/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = gccas
USEDLIBS = LLVMAsmParser.a LLVMBCWriter.a LLVMTransforms.a LLVMipo.a LLVMipa.a \
           LLVMScalarOpts.a LLVMAnalysis.a LLVMTarget.a LLVMTransformUtils.a \
           LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
