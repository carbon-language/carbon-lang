#===-- tools/analyze/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = analyze

analyze_SOURCES = \
  AnalysisWrappers.cpp \
  analyze.cpp \
  GraphPrinters.cpp

analyze_LDADD = \
  $(call GETOBJS,AsmParser,BCReader,Analysis,IPA,DataStructure,ScalarOpts) \
  $(call GETLIBS,Transforms,Target,ScalarOpts,TransformUtils) \
  $(call GETOBJS,Core) \
  $(call GETLIBS,Support,System)
