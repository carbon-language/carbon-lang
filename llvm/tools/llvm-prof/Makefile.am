#===-- tools/llvm-prof/Makefile.am -------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llvm-prof

llvm_prof_SOURCES = llvm-prof.cpp
llvm_prof_LDADD = \
  $(call GETLIBS,Analysis) \
  $(call GETOBJS,BCReader,Core) \
  $(call GETLIBS,Support,System)
