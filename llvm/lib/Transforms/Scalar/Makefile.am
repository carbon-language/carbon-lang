#===-- lib/Transforms/Scalar/Makefile.am -------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMScalarOpts.a
PRELINK=libLLVMScalarOpts.a

MYSOURCES = \
  ADCE.cpp \
  BasicBlockPlacement.cpp \
  ConstantProp.cpp \
  CorrelatedExprs.cpp \
  DCE.cpp \
  DeadStoreElimination.cpp \
  DecomposeMultiDimRefs.cpp \
  GCSE.cpp \
  IndVarSimplify.cpp \
  InstructionCombining.cpp \
  LICM.cpp \
  LoopSimplify.cpp \
  LoopUnroll.cpp \
  LoopUnswitch.cpp \
  LowerAllocations.cpp \
  LowerConstantExprs.cpp \
  LowerGC.cpp \
  LowerInvoke.cpp \
  LowerPacked.cpp \
  LowerSelect.cpp \
  LowerSwitch.cpp \
  Mem2Reg.cpp \
  PRE.cpp \
  Reassociate.cpp \
  ScalarReplAggregates.cpp \
  SCCP.cpp \
  SimplifyCFG.cpp \
  SymbolStripping.cpp \
  TailDuplication.cpp \
  TailRecursionElimination.cpp

libLLVMScalarOpts_a_SOURCES = $(MYSOURCES)

