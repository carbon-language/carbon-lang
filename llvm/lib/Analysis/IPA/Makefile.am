#===-- lib/Analysis/IPA/Makefile.am ------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

lib_LIBRARIES = libLLVMIPA.a
libexec_PROGRAMS = LLVMIPA.o

MYSOURCES = \
  Andersens.cpp \
  CallGraph.cpp \
  CallGraphSCCPass.cpp \
  FindUnsafePointerTypes.cpp \
  FindUsedTypes.cpp \
  GlobalsModRef.cpp \
  PrintSCC.cpp
libLLVMIPA_a_SOURCES = $(MYSOURCES)
LLVMIPA_o_SOURCES = $(MYSOURCES)
LIBS=
