#===-- lib/Target/Skeleton/Makefile.am ---------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#


include $(top_srcdir)/Makefile.rules.am

libexec_PROGRAMS = LLVMSkeleton.o

BUILT_SOURCES = \
  SkeletonGenRegisterInfo.h.inc \
  SkeletonGenRegisterNames.inc \
  SkeletonGenRegisterInfo.inc \
  SkeletonGenInstrNames.inc \
  SkeletonGenInstrInfo.inc

LLVMSkeleton_o_SOURCES = \
  SkeletonInstrInfo.cpp \
  SkeletonJITInfo.cpp \
  SkeletonRegisterInfo.cpp \
  SkeletonTargetMachine.cpp \
  $(BUILT_SOURCES)

LIBS=

$(BUILT_SOURCES) : $(TDFILES) $(TBLGEN)
