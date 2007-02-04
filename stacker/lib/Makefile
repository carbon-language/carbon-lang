#===-- projects/llvm-stacker/lib/Makefile -------------------*- Makefile -*-===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#
#
# Compile Stacker libraries
#
#===------------------------------------------------------------------------===#

LEVEL = ..
DIRS = compiler runtime

include $(LEVEL)/Makefile.common

# Don't generate the runtime if we don't have LLVMGCC
ifeq ($(strip $(LLVMGCC)),)
  DIRS := $(filter-out runtime, $(DIRS))
endif
