#===-- lib/Target/SparcV9/RegAlloc/Makefile.am -------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

lib_LIBRARIES = libLLVMSparcV9RegAlloc.a
libexec_PROGRAMS = LLVMSparcV9RegAlloc.o

MYSOURCES = \
  IGNode.cpp \
  InterferenceGraph.cpp \
  LiveRangeInfo.cpp \
  PhyRegAlloc.cpp \
  RegClass.cpp

libLLVMSparcV9RegAlloc_a_SOURCES = $(MYSOURCES)
LLVMSparcV9RegAlloc_o_SOURCES = $(MYSOURCES)
LIBS=
