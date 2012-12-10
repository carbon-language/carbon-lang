##===- source/Utility/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbUtility
BUILD_ARCHIVE = 1
NO_PEDANTIC = 1

# Enable RTTI on GCC builds because liblldbCore.a requires RTTI.
# See source/Core/Makefile for details.
ifeq (g++,$(shell basename $(CXX)))
  REQUIRES_RTTI = 1
endif


include $(LLDB_LEVEL)/Makefile
