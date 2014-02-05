##===- source/Makefile -------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ..
include $(LLDB_LEVEL)/../../Makefile.config

# enable lldb-gdbserver for supported platforms
DIRS :=
ifneq (,$(strip $(filter $(HOST_OS), FreeBSD Linux)))
ifneq (,$(strip $(filter $(HOST_ARCH), x86_64)))
DIRS += lldb-gdbserver
else
endif
else
endif

ifneq ($(HOST_OS),MingW)
DIRS += driver lldb-platform
endif

ifeq ($(HOST_OS),Darwin)
DIRS += debugserver
endif

include $(LLDB_LEVEL)/Makefile
