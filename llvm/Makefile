#===- ./Makefile -------------------------------------------*- Makefile -*--===#
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#
LEVEL = .
DIRS = lib/System lib/Support utils lib tools 

ifneq ($(MAKECMDGOALS),tools-only)
DIRS += runtime
OPTIONAL_DIRS = examples projects
endif

include $(LEVEL)/Makefile.common

test :: all
	cd test; $(MAKE)

tools-only: all
