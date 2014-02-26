##===- scripts/Python/modules/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../..
include $(LLDB_LEVEL)/../../Makefile.config

DIRS:=

# only build the readline suppression module on Linux
ifeq ($(HOST_OS), Linux)
DIRS += readline
endif

include $(LLDB_LEVEL)/Makefile
