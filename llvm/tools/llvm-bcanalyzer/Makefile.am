#===-- tools/llvm-bcanalyzer/Makefile.am -------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llvm-bcanalyzer

llvm_bcanalyzer_SOURCES = llvm-bcanalyzer.cpp

llvm_bcanalyzer_LDADD = \
  $(call GETOBJS,BCReader,Core) \
  $(call GETLIBS,Support,System)
