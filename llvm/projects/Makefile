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

DIRS=
OPTIONAL_DIRS:=$(basename $(shell ls $(SourceDir) | grep -v CVS | grep -v Makefile))

include $(BUILD_SRC_ROOT)/Makefile.rules
