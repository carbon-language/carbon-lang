#===-- tools/llvm-nm/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llvm-nm

llvm_nm_SOURCES = llvm-nm.cpp
llvm_nm_LDADD = \
  $(call GETOBJS,BCReader,Core) \
  $(call GETLIBS,Support,System)
