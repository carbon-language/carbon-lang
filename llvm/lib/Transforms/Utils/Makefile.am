#===-- lib/Transforms/Utils/Makefile.am --------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMTransformUtils.a

MYSOURCES = \
  BasicBlockUtils.cpp \
  BreakCriticalEdges.cpp \
  CloneFunction.cpp \
  CloneModule.cpp \
  CloneTrace.cpp \
  CodeExtractor.cpp \
  DemoteRegToStack.cpp \
  InlineFunction.cpp \
  Local.cpp \
  PromoteMemoryToRegister.cpp \
  SimplifyCFG.cpp \
  UnifyFunctionExitNodes.cpp \
  ValueMapper.cpp

libLLVMTransformUtils_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMTransformUtils.a
