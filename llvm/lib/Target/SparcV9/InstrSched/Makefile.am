#===-- lib/Target/SparcV9/InstrSched/Makefile.am -----------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

lib_LIBRARIES = libLLVMSparcV9InstrSched.a
libexec_PROGRAMS = LLVMSparcV9InstrSched.o

MYSOURCES = \
  InstrScheduling.cpp \
  SchedGraphCommon.cpp \
  SchedGraph.cpp \
  SchedPriorities.cpp

libLLVMSparcV9InstrSched_a_SOURCES = $(MYSOURCES)
LLVMSparcV9InstrSched_o_SOURCES = $(MYSOURCES)
LIBS=
