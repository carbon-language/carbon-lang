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

# tfiala test commit: will include lldb-gdbserver for linux x86_64 soon.
ifneq ($(HOST_OS),MingW)
DIRS := driver lldb-platform
endif

include $(LLDB_LEVEL)/Makefile
