#===-- tools/analyze/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = analyze

analyze_SOURCES = \
  AnalysisWrappers.cpp \
  analyze.cpp \
  GraphPrinters.cpp

#USEDLIBS = asmparser bcreader analysis ipa datastructure scalaropts.a transforms.a \
#           target.a scalaropts.a transformutils.a vmcore support LLVMsystem.a

analyze_LDADD = \
  $(call GETOBJS,AsmParser,BCReader,Core) \
  $(call GETLIBS,Analysis,IPA,DataStructure,ScalarOpts,Transforms,Target) \
  $(call GETLIBS,ScalarOpts,TransformUtils,Support,System)
