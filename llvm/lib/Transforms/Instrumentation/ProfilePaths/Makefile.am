#===- lib/Transforms/Instrumentation/ProfilePaths/Makefile.am -*- Makefile -*--#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMProfilePaths.a

MYSOURCES= \
  CombineBranch.cpp \
  EdgeCode.cpp \
  GraphAuxiliary.cpp \
  Graph.cpp \
  InstLoops.cpp \
  ProfilePaths.cpp \
  RetracePath.cpp

libLLVMProfilePaths_a_SOURCES = $(MYSOURCES)

PRELINK=libLLVMProfilePaths.a
