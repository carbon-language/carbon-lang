#===- tools/llc/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llc
USEDLIBS = \
	LLVMCBackend \
	LLVMPowerPC \
	LLVMSparcV9 \
	LLVMX86 \
	LLVMSkeleton \
	LLVMSelectionDAG \
	LLVMSparcV9RegAlloc \
	LLVMSparcV9InstrSched \
	LLVMCodeGen \
	LLVMTarget.a \
	LLVMSparcV9LiveVar \
	LLVMipa.a \
	LLVMTransforms.a \
	LLVMScalarOpts.a \
	LLVMAnalysis.a \
	LLVMTransformUtils.a \
	LLVMBCReader \
	LLVMBCWriter \
	LLVMCore \
	LLVMSupport.a \
	LLVMSystem.a

TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common

