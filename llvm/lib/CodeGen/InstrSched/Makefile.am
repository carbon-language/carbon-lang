#===-- lib/CodeGen/InstrSched/Makefile.am ------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

lib_LIBRARIES = libLLVMInstrSched.a
libexec_PROGRAMS = LLVMInstrSched.o

MYSOURCES = \
  InstrScheduling.cpp \
  SchedGraphCommon.cpp \
  SchedGraph.cpp \
  SchedPriorities.cpp

libLLVMInstrSched_a_SOURCES = $(MYSOURCES)
LLVMInstrSched_o_SOURCES = $(MYSOURCES)
LIBS=
