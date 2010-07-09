##===- source/Host/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..

include $(LLDB_LEVEL)/../../Makefile.config

ifeq ($(HOST_OS),Darwin)
DIRS := macosx posix
else
DIRS := linux posix
endif

include $(LLDB_LEVEL)/Makefile
