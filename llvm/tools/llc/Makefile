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

USEDLIBS := $(addprefix LLVM,$(TARGETS_TO_BUILD)) \
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

