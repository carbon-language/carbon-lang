#===- lib/Transforms/Instrumentation/ProfilePaths/Makefile.am -*- Makefile -*--#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

lib_LIBRARIES = libLLVMProfilePaths.a
libexec_PROGRAMS = LLVMProfilePaths.o

MYSOURCES= \
  CombineBranch.cpp \
  EdgeCode.cpp \
  GraphAuxiliary.cpp \
  Graph.cpp \
  InstLoops.cpp \
  ProfilePaths.cpp \
  RetracePath.cpp

libLLVMProfilePaths_a_SOURCES = $(MYSOURCES)
LLVMProfilePaths_o_SOURCES = $(MYSOURCES)
LIBS=
