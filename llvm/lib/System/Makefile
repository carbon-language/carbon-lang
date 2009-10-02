##===- lib/System/Makefile ---------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMSystem
BUILD_ARCHIVE = 1

include $(LEVEL)/Makefile.config

ifeq ($(HOST_OS),MingW)
  REQUIRES_EH := 1
endif

EXTRA_DIST = Unix Win32 README.txt

include $(LEVEL)/Makefile.common

CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))
