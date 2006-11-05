##===- projects/Stacker/lib/Makefile -----------------------*- Makefile -*-===##
#
# Compile Stacker libraries
#
##===----------------------------------------------------------------------===##

LEVEL = ..
DIRS = compiler runtime

include $(LEVEL)/Makefile.common

# Don't generate the runtime if we don't have LLVMGCC
ifeq ($(strip $(LLVMGCC)),)
  DIRS := $(filter-out runtime, $(DIRS))
endif

