##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..

include $(CLANG_LEVEL)/../../Makefile.config

# Build libclang first so that other tools can depend on it.
DIRS := libclang

PARALLEL_DIRS := driver c-index-test arcmt-test c-arcmt-test diagtool \
								 clang-check

# Recurse into the extra repository of tools if present.
PARALLEL_OPTIONAL_DIRS := extra

include $(CLANG_LEVEL)/Makefile
