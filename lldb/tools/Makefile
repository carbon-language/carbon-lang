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

DIRS :=

# enable lldb-gdbserver for supported platforms
ifneq (,$(strip $(filter $(HOST_OS), FreeBSD Linux GNU/kFreeBSD)))
DIRS += lldb-gdbserver
endif

ifneq ($(HOST_OS),MingW)
DIRS += lldb-platform
endif

ifeq ($(HOST_OS),Darwin)
DIRS += debugserver
endif

DIRS += lldb-mi
DIRS += driver

include $(LLDB_LEVEL)/Makefile
