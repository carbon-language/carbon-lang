#===-- tools/gccas/Makefile.am -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = gccas

gccas_SOURCES = gccas.cpp

gccas_LDADD = \
  $(call GETOBJS,AsmParser,BCWriter,Core) \
  $(call GETLIBS,Transforms,IPO,IPA,ScalarOpts,Analysis,Target,TransformUtils) \
  $(call GETLIBS,Support,System)
