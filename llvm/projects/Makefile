##===- projects/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL=..

include $(LEVEL)/Makefile.config

DIRS:= $(filter-out llvm-test,$(patsubst $(SourceDir)/%/Makefile,%,$(wildcard $(SourceDir)/*/Makefile)))

# Sparc cannot link shared libraries (libtool problem?) which Stacker uses
ifeq ($(ARCH), Sparc)
DIRS := $(filter-out Stacker, $(DIRS))
endif

include $(BUILD_SRC_ROOT)/Makefile.rules
