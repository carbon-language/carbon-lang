#===-- tools/llc/Makefile.am -------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llc

llc_SOURCES = llc.cpp

llc_LDADD = \
  $(call GETOBJS,CWriter,PowerPC,SparcV9,X86,Skeleton,SelectionDAG) \
  $(call GETOBJS,SparcV9RegAlloc,SparcV9InstrSched,CodeGen) \
  $(call GETLIBS,Target) \
  $(call GETOBJS,SparcV9LiveVar) \
  $(call GETLIBS,IPA,Transforms,ScalarOpts,Analysis,TransformUtils) \
  $(call GETOBJS,BCReader,BCWriter,Core) \
  $(call GETLIBS,Support,System)
