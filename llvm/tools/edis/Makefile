##===- tools/ed/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = EnhancedDisassembly
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/EnhancedDisassembly.exports

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := mcdisassembler

# If the X86 target is enabled, link in the asmprinter and disassembler.
ifneq ($(filter $(TARGETS_TO_BUILD), X86),)
LINK_COMPONENTS += x86asmprinter x86disassembler
endif

# If the X86 target is enabled, link in the asmprinter and disassembler.
ifneq ($(filter $(TARGETS_TO_BUILD), ARM),)
LINK_COMPONENTS += armasmprinter armdisassembler
endif

include $(LEVEL)/Makefile.common

ifeq ($(HOST_OS),Darwin)
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -Wl,-dead_strip

    ifdef EDIS_VERSION
        LLVMLibsOptions    := $(LLVMLibsOptions) -Wl,-current_version -Wl,$(EDIS_VERSION) \
                              -Wl,-compatibility_version -Wl,1
    endif

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions)  \
                            -Wl,-install_name \
                            -Wl,"@rpath/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

