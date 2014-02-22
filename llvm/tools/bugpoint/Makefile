##===- tools/bugpoint/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := bugpoint
LINK_COMPONENTS := asmparser instrumentation scalaropts ipo linker bitreader \
                   bitwriter irreader vectorize objcarcopts codegen

# Support plugins.
NO_DEAD_STRIP := 1

include $(LEVEL)/Makefile.common
