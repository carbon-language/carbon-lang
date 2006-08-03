##===- tools/lto/Makefile ----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Devang Patel and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMlto

LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1
LOADABLE_MODULE = 1
DONT_BUILD_RELINKED = 1

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up USEDLIBS properly before includeing Makefile.rules
include $(LEVEL)/Makefile.config

USEDLIBS := $(addprefix LLVM,$(TARGETS_TO_BUILD)) \
	LLVMSelectionDAG.a LLVMCodeGen.a LLVMipo.a LLVMTransforms.a \
	LLVMScalarOpts.a LLVMipa.a LLVMTransformUtils.a LLVMAnalysis.a \
	LLVMTarget.a LLVMBCReader.a LLVMBCWriter.a LLVMSystem.a LLVMLinker.a \
	LLVMCore.a LLVMSupport.a LLVMbzip2.a

include $(LEVEL)/Makefile.common

