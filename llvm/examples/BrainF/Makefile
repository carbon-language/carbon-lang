##===- examples/BrainF/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = BrainF
EXAMPLE_TOOL = 1
# To keep the shared library working, we link a few of the examples
# against it unconditionally.
ENABLE_SHARED = 1

LINK_COMPONENTS := jit bitwriter nativecodegen interpreter

include $(LEVEL)/Makefile.common
