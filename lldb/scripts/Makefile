##===- scripts/Makefile ------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ..
include $(LLDB_LEVEL)/../../Makefile.config

ifeq (,$(findstring -DLLDB_DISABLE_PYTHON,$(CXXFLAGS)))
DIRS := Python
endif

include $(LLDB_LEVEL)/Makefile
