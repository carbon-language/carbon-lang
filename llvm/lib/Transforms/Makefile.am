#===-- lib/Transforms/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

SUBDIRS = Utils Instrumentation Scalar IPO

lib_LIBRARIES = libLLVMTransforms.a

MYSOURCES = ExprTypeConvert.cpp LevelRaise.cpp TransformInternals.cpp

libLLVMTransforms_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMTransforms.a
