##===- lib/Makefile ----------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ..

include $(LEVEL)/Makefile.config

PARALLEL_DIRS := VMCore AsmParser Bitcode Archive Analysis Transforms CodeGen \
                Target ExecutionEngine Debugger Linker MC

# Only build the CompilerDriver when PIC is enabled.

ifeq ($(ENABLE_PIC),1)
PARALLEL_DIRS += CompilerDriver
endif

include $(LEVEL)/Makefile.common

