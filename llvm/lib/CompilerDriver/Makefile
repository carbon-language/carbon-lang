##===- lib/CompilerDriver/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..

# We don't want this library to appear in `llvm-config --libs` output, so its
# name doesn't start with "LLVM".

LIBRARYNAME = CompilerDriver
LINK_COMPONENTS = support system

include $(LEVEL)/Makefile.common
