##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..
DIRS := driver libclang c-index-test

include $(CLANG_LEVEL)/../../Makefile.config

ifeq ($(OS), $(filter $(OS), Cygwin MingW Minix))
DIRS := $(filter-out libclang c-index-test, $(DIRS))
endif

include $(CLANG_LEVEL)/Makefile
