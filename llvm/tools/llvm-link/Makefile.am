#===-- tools/llvm-link/Makefile.am -------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = llvm-link

llvm_link_SOURCES = llvm-link.cpp
llvm_link_LDADD = \
  $(call GETOBJS,BCReader,BCWriter,Core) \
  $(call GETLIBS,Support,System)
