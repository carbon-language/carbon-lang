##===- runtime/libprofile/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
include $(LEVEL)/Makefile.config

ifneq ($(strip $(LLVMCC)),)
BYTECODE_LIBRARY = 1
endif
LIBRARYNAME = profile_rt
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1
EXTRA_DIST = libprofile.exports
EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/libprofile.exports

include $(LEVEL)/Makefile.common
