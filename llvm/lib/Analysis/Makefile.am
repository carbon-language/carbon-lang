#===-- lib/Analysis/Makefile.am ----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

SUBDIRS = IPA DataStructure

lib_LIBRARIES = libLLVMAnalysis.a
libexec_PROGRAMS = LLVMAnalysis.o

MYSOURCES = \
  AliasAnalysisCounter.cpp \
  AliasAnalysis.cpp \
  AliasAnalysisEvaluator.cpp \
  AliasSetTracker.cpp \
  BasicAliasAnalysis.cpp \
  CFGPrinter.cpp \
  Expressions.cpp \
  InstCount.cpp \
  Interval.cpp \
  IntervalPartition.cpp \
  LoadValueNumbering.cpp \
  LoopInfo.cpp \
  PostDominators.cpp \
  ProfileInfo.cpp \
  ProfileInfoLoader.cpp \
  ProfileInfoLoaderPass.cpp \
  ScalarEvolution.cpp \
  Trace.cpp \
  ValueNumbering.cpp

libLLVMAnalysis_a_SOURCES = $(MYSOURCES)
LLVMAnalysis_o_SOURCES = $(MYSOURCES)
LIBS=
