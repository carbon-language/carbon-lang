#===-- tools/gccld/Makefile.am -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = gccld

gccld_SOURCES = gccld.cpp GenerateCode.cpp Linker.cpp

gccld_LDADD = \
  $(call GETLIBS,IPO,Transforms,ScalarOpts,Analysis,IPA,TransformUtils,Target) \
  $(call GETOBJS,BCReader,BCWriter,Core) \
  $(call GETLIBS,Support,System)
