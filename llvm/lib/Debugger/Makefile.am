#===-- lib/Debugger/Makefile.am ----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMDebugger.a

libLLVMDebugger_a_SOURCES = \
  Debugger.cpp \
  ProgramInfo.cpp \
  RuntimeInfo.cpp \
  SourceFile.cpp \
  SourceLanguage-CFamily.cpp \
  SourceLanguage-CPlusPlus.cpp \
  SourceLanguage.cpp \
  SourceLanguage-Unknown.cpp \
  UnixLocalInferiorProcess.cpp

PRELINK=libLLVMDebugger.a
