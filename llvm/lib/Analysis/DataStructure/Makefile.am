#===-- lib/Analysis/DataStructure/Makefile.am --------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMDataStructure.a

MYSOURCES = \
  BottomUpClosure.cpp\
  CompleteBottomUp.cpp\
  DataStructureAA.cpp\
  DataStructure.cpp\
  DataStructureOpt.cpp\
  DataStructureStats.cpp\
  DependenceGraph.cpp\
  GraphChecker.cpp\
  IPModRef.cpp\
  Local.cpp\
  MemoryDepAnalysis.cpp\
  Parallelize.cpp\
  PgmDependenceGraph.cpp\
  Printer.cpp\
  Steensgaard.cpp\
  TopDownClosure.cpp

libLLVMDataStructure_a_SOURCES = $(MYSOURCES)
PRELINK=libLLVMDataStructure.a
