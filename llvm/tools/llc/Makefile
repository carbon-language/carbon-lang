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
	LLVMSparcV9ModuloSched \
	LLVMCBackend \
	LLVMPowerPC \
	LLVMSparcV8 \
	LLVMSparcV9 \
	LLVMX86 \
	LLVMAlpha \
	LLVMSkeleton \
	LLVMSparcV9RegAlloc \
	LLVMSparcV9InstrSched \
	LLVMSelectionDAG \
	LLVMCodeGen \
	LLVMTarget.a \
	LLVMSparcV9LiveVar \
	LLVMipa.a \
	LLVMTransforms.a \
	LLVMScalarOpts \
	LLVMAnalysis.a \
	LLVMTransformUtils.a \
	LLVMBCReader \
	LLVMBCWriter \
	LLVMCore \
	LLVMSupport.a \
	LLVMbzip2 \
	LLVMSystem.a

TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common

