##===- tools/shlib/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

LIBRARYNAME = LLVM-$(LLVMVersion)

NO_BUILD_ARCHIVE = 1
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

include $(LEVEL)/Makefile.common

# Include all archives in libLLVM.(so|dylib) except the ones that have
# their own dynamic libraries.
Archives := $(wildcard $(LibDir)/libLLVM*.a)
SharedLibraries := $(wildcard $(LibDir)/libLLVM*$(SHLIBEXT))
IncludeInLibLlvm := $(filter-out $(basename $(SharedLibraries)).a, $(Archives))
LLVMLibsOptions := $(IncludeInLibLlvm:$(LibDir)/lib%.a=-l%)
LLVMLibsPaths   := $(IncludeInLibLlvm)

$(LibName.SO): $(LLVMLibsPaths)

ifeq ($(HOST_OS),Darwin)
    # set dylib internal version number to llvmCore submission number
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
                        -Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION) \
                        -Wl,-compatibility_version -Wl,1
    endif
    # Include everything from the .a's into the shared library.
    LLVMLibsOptions    := $(LLVMLibsOptions) -all_load
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -avoid-version \
                         -Wl,-dead_strip \
                         -Wl,-seg1addr -Wl,0xE0000000 

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions)  \
                            -Wl,-install_name \
                            -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

ifeq ($(HOST_OS), Linux)
    # Include everything from the .a's into the shared library.
    LLVMLibsOptions := -Wl,--whole-archive $(LLVMLibsOptions) \
                       -Wl,--no-whole-archive
    # Don't allow unresolved symbols.
    LLVMLibsOptions += -Wl,--no-undefined
endif
