#===-- lib/ExecutionEngine/Makefile.am ---------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

SUBDIRS = Interpreter JIT

lib_LIBRARIES = libLLVMExecutionEngine.a

libLLVMExecutionEngine_a_SOURCES = ExecutionEngine.cpp

PRELINK=libLLVMExecutionEngine.a
