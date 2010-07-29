##===- unittests/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ..

PARALLEL_DIRS = ADT ExecutionEngine Support Transforms VMCore

include $(LEVEL)/Makefile.common

clean::
	$(Verb) $(RM) -rf Analysis
	$(Verb) $(RM) -f *Tests
