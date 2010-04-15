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

BUILT_SOURCES = EDInfo.inc

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/EnhancedDisassembly.exports

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

ifeq ($(ENABLE_PIC),1)
    ifneq ($(TARGET_OS), $(filter $(TARGET_OS), Cygwin MingW))
	LINK_LIBS_IN_SHARED = 1
	SHARED_LIBRARY = 1
    endif
endif

LINK_COMPONENTS := $(TARGETS_TO_BUILD) x86asmprinter x86disassembler

include $(LEVEL)/Makefile.common

ifeq ($(HOST_OS),Darwin)
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -avoid-version \
                         -Wl,-dead_strip

    ifdef EDIS_VERSION
        LLVMLibsOptions    := $(LLVMLibsOptions) -Wl,-current_version -Wl,$(EDIS_VERSION) \
                              -Wl,-compatibility_version -Wl,1
    endif

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions)  \
                            -no-undefined -Wl,-install_name \
                            -Wl,"@rpath/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

EDInfo.inc:	$(TBLGEN)
	$(Echo) "Building semantic information header"
	$(Verb) $(TableGen) -o $(call SYSPATH, $@) -gen-enhanced-disassembly-header /dev/null

clean::
	-$(Verb) $(RM) -f EDInfo.inc
