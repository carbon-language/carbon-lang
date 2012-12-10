##===- source/Core/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbCore
BUILD_ARCHIVE = 1

# Enable RTTI on GCC builds because one source file in this directory
# (cxa_demangle.cpp) uses dynamic_cast<> and GCC (at least 4.6 and 4.7)
# complain if we try to compile it with -fno-rtti. This is somewhat of a
# kludge because it forces us to enable RTTI in liblldbUtility.a and also
# link in additional clang static libraries to resolve vtable references,
# but actually has negligible impact on (shard object) file size.
$(info shell basename CXX is $(shell basename $(CXX)))
ifeq (g++,$(shell basename $(CXX)))
  REQUIRES_RTTI = 1
endif

include $(LLDB_LEVEL)/Makefile
