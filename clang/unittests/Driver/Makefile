##===- unittests/Driver/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = Multilib
include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) support option
USEDLIBS = clangDriver.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
