##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LLDB_LEVEL := ../..

TOOLNAME = lldb

LD.Flags += -ledit -llldb -llldbUtility

# # Include this here so we can get the configuration of the targets that have
# # been configured for construction. We have to do this early so we can set up
# # LINK_COMPONENTS before including Makefile.rules
include $(LLDB_LEVEL)/../../Makefile.config

ifeq ($(HOST_OS),Darwin)
	LD.Flags += -Wl,-rpath,@loader_path/../lib/
	LD.Flags += -Wl,-sectcreate -Wl,__TEXT -Wl,__info_plist -Wl,lldb-Info.plist
endif

ifeq ($(HOST_OS),Linux)
	LD.Flags += -Wl,-rpath,$(LibDir)
endif

include $(LLDB_LEVEL)/Makefile
