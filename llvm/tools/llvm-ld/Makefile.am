#===-- tools/llvm-ld/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = llvm-ld

llvm_ld_SOURCES = GenerateCode.cpp Linker.cpp llvm-ld.cpp
llvm_ld_LDADD = \
  $(call GETLIBS,IPO,Transforms,ScalarOpts,Analysis,IPA,TransformUtils,Target) \
  $(call GETOBJS,BCReader,BCWriter,Core) \
  $(call GETLIBS,Support,System)
