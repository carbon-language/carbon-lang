##===- clang/runtime/libcxx/Makefile -----------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This file defines support for building the "libc++" C++ standard library as
# part of a Clang compiler build.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
LEVEL := $(CLANG_LEVEL)/../..
LIBRARYNAME = c++

LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

# Include LLVM common makefile.
include $(LEVEL)/Makefile.config

# Expect libcxx to be in llvm/projects/libcxx
LIBCXX_SRC_ROOT := $(LLVM_SRC_ROOT)/projects/libcxx

# Override the source root to point at the libcxx sources.
PROJ_MAKEFILE := $(PROJ_SRC_DIR)/Makefile
PROJ_SRC_DIR := $(LIBCXX_SRC_ROOT)/src
CPP.Flags := -nostdinc++ -I$(LIBCXX_SRC_ROOT)/include
CXX.Flags := -std=c++0x

# Include LLVM makefile rules.
include $(LLVM_SRC_ROOT)/Makefile.rules

# Force building with the just built Clang.
#
# FIXME: Do we really want to do this? It is uber slow.
# CXX := $(ToolDir)/clang

ifeq ($(HOST_OS),Darwin)
    LLVMLibsOptions += -Wl,-compatibility_version,1

    # Don't link with default libraries.
    LLVMLibsOptions += -nodefaultlibs -lSystem

    # Reexport libc++abi.
    LLVMLibsOptions += -Wl,-reexport_library,/usr/lib/libc++abi.dylib

    # Set dylib internal version number to submission number.
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions += -Wl,-current_version \
                           -Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION)
    endif

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions += -Wl,-install_name \
                          -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif
