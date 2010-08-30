##===- examples/TracingBrainF/Makefile ------------------- -*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = TracingBrainF
EXAMPLE_TOOL = 1

CXXFLAGS += -foptimize-sibling-calls

LINK_COMPONENTS := scalaropts jit nativecodegen

include $(LEVEL)/Makefile.common
