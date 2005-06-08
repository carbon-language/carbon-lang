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

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up USEDLIBS properly before includeing Makefile.rules
include $(LEVEL)/Makefile.config

# Initialize the USEDLIBS so we can add to it
USEDLIBS :=

# Check for LLVMCBackend  target
ifneq ($(strip $(filter CBackend,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMCBackend
endif

# Check for Skeleton target
ifneq ($(strip $(filter Skeleton,$(TARGETS_TO_BUILD))),)
USEDLIB += LLVMSkeleton
endif

# Check for Sparc target
ifneq ($(strip $(filter SparcV9,$(TARGETS_TO_BUILD))),)
USEDLIBS += \
	LLVMSparcV9ModuloSched \
	LLVMSparcV9 \
	LLVMSparcV9RegAlloc \
	LLVMSparcV9InstrSched \
	LLVMSparcV9LiveVar
endif

ifneq ($(strip $(filter SparcV8,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMSparcV8
endif


#Check for X86 Target
ifneq ($(strip $(filter X86,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMX86
endif

#Check for PowerPC Target
ifneq ($(strip $(filter PowerPC,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMPowerPC
endif

#Check for Alpha Target
ifneq ($(strip $(filter Alpha,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMAlpha
endif

#Check for IA64 Target
ifneq ($(strip $(filter IA64,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMIA64
endif

USEDLIBS += \
	LLVMSelectionDAG \
	LLVMCodeGen \
	LLVMTarget.a \
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

include $(LLVM_SRC_ROOT)/Makefile.rules

