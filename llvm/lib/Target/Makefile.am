#===-- lib/Target/Makefile.am ------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

SUBDIRS = X86 CBackend PowerPC SparcV9 Skeleton 

lib_LIBRARIES = libLLVMTarget.a

libLLVMTarget_a_SOURCES = \
  MRegisterInfo.cpp \
  TargetData.cpp \
  TargetFrameInfo.cpp \
  TargetInstrInfo.cpp \
  TargetMachine.cpp \
  TargetMachineRegistry.cpp \
  TargetSchedInfo.cpp
