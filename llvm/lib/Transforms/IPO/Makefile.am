#===-- lib/Transforms/IPO/Makefile.am ----------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMIPO.a

MYSOURCES= \
  ArgumentPromotion.cpp \
  ConstantMerge.cpp \
  DeadArgumentElimination.cpp \
  DeadTypeElimination.cpp \
  ExtractFunction.cpp \
  FunctionResolution.cpp \
  GlobalDCE.cpp \
  GlobalOpt.cpp \
  Inliner.cpp \
  InlineSimple.cpp \
  Internalize.cpp \
  IPConstantPropagation.cpp \
  LoopExtractor.cpp \
  LowerSetJmp.cpp \
  PruneEH.cpp \
  RaiseAllocations.cpp

libLLVMIPO_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMIPO.a
