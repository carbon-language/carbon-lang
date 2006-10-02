##===- projects/Stacker/lib/Makefile -----------------------*- Makefile -*-===##
#
# Compile Stacker libraries
#
##===----------------------------------------------------------------------===##

LEVEL = ..
DIRS = compiler runtime

# Don't generate the runtime if we don't have LLVMGCC
ifeq ($(LLVMGCC),)
  DIRS := $(filter-out runtime, $(DIRS))
endif

include $(LEVEL)/Makefile.common
