##===- tools/llvm-db/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llvm-db
USEDLIBS = LLVMDebugger

# Enable JIT support
include ../Makefile.JIT

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common

