#===-- lib/Target/SparcV9/LiveVar/Makefile.am --------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMSparcV9LiveVar.a

MYSOURCES = \
  BBLiveVar.cpp \
  FunctionLiveVarInfo.cpp \
  ValueSet.cpp

libLLVMSparcV9LiveVar_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMSparcV9LiveVar.a
