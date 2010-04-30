##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../../..
DIRS := driver libclang c-index-test

include $(LEVEL)/Makefile.config

ifeq ($(OS), $(filter $(OS), Cygwin MingW))
DIRS := $(filter-out libclang c-index-test, $(DIRS))
endif

include $(LEVEL)/Makefile.common
