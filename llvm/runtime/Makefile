##===- runtime/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
# llvm/runtime Makefile:  Build all subdirectories automatically

LEVEL = ..
include $(LEVEL)/Makefile.config
ifneq ($(wildcard $(LLVMGCCDIR)),)
PARALLEL_DIRS  := $(sort $(filter-out Makefile, $(filter-out README.txt, $(filter-out Output, $(filter-out CVS, $(notdir $(wildcard $(SourceDir)/*)))))))
else
PARALLEL_DIRS  := 
all ::
	@echo '********' Warning: Your LLVMGCCDIR is set incorrectly.  Double-check 
	@echo '********' Warning: llvm/Makefile.config to make sure it matches
	@echo '********' Warning: the directory where the C front-end is installed,
	@echo '********' Warning: and re-run configure if it does not.
endif

include $(LEVEL)/Makefile.common

# Install target for libraries: Copy into the gcc install directory in chris's
# tree...
#
install::

clean::
