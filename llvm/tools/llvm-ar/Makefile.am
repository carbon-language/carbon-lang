#===-- tools/llvm-ar/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = llvm-ar

llvm_ar_SOURCES = llvm-ar.cpp
llvm_ar_LDADD = \
  $(call GETOBJS,BCReader,Core) \
  $(call GETLIBS,Support,System)
