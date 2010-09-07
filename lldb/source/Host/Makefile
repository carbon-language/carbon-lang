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

DIRS := common

ifeq ($(HOST_OS),Darwin)
DIRS += macosx
endif

include $(LLDB_LEVEL)/Makefile
