#===-- tools/opt/Makefile.am -------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = opt

opt_SOURCES = opt.cpp

opt_LDADD = \
  $(call GETOBJS,BCReader,BCWriter,Instrument,ProfilePaths,ScalarOpts) \
  $(call GETOBJS,IPO,IPA,DataStructure,Transforms) \
  $(call GETLIBS,Target) \
  $(call GETOBJS,Analysis,TransformUtils,Core) \
  $(call GETLIBS,Support,System) 
