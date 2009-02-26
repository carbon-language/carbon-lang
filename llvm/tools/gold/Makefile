#===- tools/gold/Makefile ----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMgold

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_LIBS_IN_SHARED=1
SHARED_LIBRARY = 1
BUILD_ARCHIVE = 0
DONT_BUILD_RELINKED = 1

LINK_COMPONENTS :=
LIBS += -llto

include $(LEVEL)/Makefile.common

# Because off_t is used in the public API, the largefile parts are required for
# ABI compatibility.
CXXFLAGS+=-I$(BINUTILS_INCDIR) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -lLTO
