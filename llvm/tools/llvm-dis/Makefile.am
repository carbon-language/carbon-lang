#===-- tools/llvm-dis/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llvm-dis

llvm_dis_SOURCES = llvm-dis.cpp
llvm_dis_LDADD = \
  $(call GETOBJS,BCReader,Core) \
  $(call GETLIBS,Support,System)
