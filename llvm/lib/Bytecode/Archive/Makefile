##===- lib/Bytecode/Archive/Makefile -----------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMArchive
REQUIRES_EH := 1

# We only want an archive so only those modules actually used by a tool are 
# included.
BUILD_ARCHIVE := 1

include $(LEVEL)/Makefile.common
