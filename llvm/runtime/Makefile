##===- runtime/Makefile ------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ..
include $(LEVEL)/Makefile.config

ifndef NO_RUNTIME_LIBS

PARALLEL_DIRS  := libprofile

ifeq ($(TARGET_OS), $(filter $(TARGET_OS), Cygwin MingW Minix))
PARALLEL_DIRS := $(filter-out libprofile, $(PARALLEL_DIRS))
endif

endif

include $(LEVEL)/Makefile.common

install::
