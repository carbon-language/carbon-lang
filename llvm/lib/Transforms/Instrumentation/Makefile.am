#===-- lib/Transforms/Instrumentation/Makefile.am ----------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

SUBDIRS = ProfilePaths

lib_LIBRARIES = libLLVMInstrument.a

MYSOURCES = \
  BlockProfiling.cpp \
  EdgeProfiling.cpp \
  EmitFunctions.cpp \
  ProfilingUtils.cpp \
  TraceBasicBlocks.cpp \
  TraceValues.cpp

libLLVMInstrument_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMInstrument.a
