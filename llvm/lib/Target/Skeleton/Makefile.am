#===-- lib/Target/Skeleton/Makefile.am ---------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#


include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMSkeleton.a

BUILT_SOURCES = \
  SkeletonGenRegisterInfo.h.inc \
  SkeletonGenRegisterNames.inc \
  SkeletonGenRegisterInfo.inc \
  SkeletonGenInstrNames.inc \
  SkeletonGenInstrInfo.inc

libLLVMSkeleton_a_SOURCES = \
  SkeletonInstrInfo.cpp \
  SkeletonJITInfo.cpp \
  SkeletonRegisterInfo.cpp \
  SkeletonTargetMachine.cpp \
  $(BUILT_SOURCES)

PRELINK=libLLVMSkeleton.a

$(BUILT_SOURCES) : $(TDFILES) $(TBLGEN)
