#===-- tools/extract/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = extract

extract_SOURCES = extract.cpp

extract_LDADD = \
  $(call GETOBJS,BCReader,BCWriter,Core) \
  $(call GETLIBS,IPO,Target,Analysis,TransformUtils,IPA,Support,System)
