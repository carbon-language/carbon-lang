#===-- lib/Target/SparcV9/InstrSched/Makefile.am -----------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMSparcV9InstrSched.a

MYSOURCES = \
  InstrScheduling.cpp \
  SchedGraphCommon.cpp \
  SchedGraph.cpp \
  SchedPriorities.cpp

libLLVMSparcV9InstrSched_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMSparcV9InstrSched.a
