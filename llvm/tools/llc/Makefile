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
REQUIRES_EH := 1

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

ifneq ($(strip $(filter Sparc,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMSparc
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

#Check for ARM Target
ifneq ($(strip $(filter ARM,$(TARGETS_TO_BUILD))),)
USEDLIBS += LLVMARM
endif

USEDLIBS += \
	LLVMCodeGen.a \
	LLVMSelectionDAG.a \
	LLVMTarget.a \
	LLVMipa.a \
	LLVMTransforms.a \
	LLVMScalarOpts.a \
	LLVMTransformUtils.a \
	LLVMAnalysis.a \
	LLVMBCReader.a \
	LLVMBCWriter.a \
	LLVMCore.a \
	LLVMSupport.a \
	LLVMbzip2.a \
	LLVMSystem.a

include $(LLVM_SRC_ROOT)/Makefile.rules

