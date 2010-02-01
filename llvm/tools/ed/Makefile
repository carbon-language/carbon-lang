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

BUILT_SOURCES = EDInfo.inc EnhancedDisassembly.exports

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

LINK_COMPONENTS := $(TARGETS_TO_BUILD) x86asmprinter x86disassembler

include $(LEVEL)/Makefile.common

ifeq ($(HOST_OS),Darwin)
    # set dylib internal version number to llvmCore submission number
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
                        -Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION) \
                        -Wl,-compatibility_version -Wl,1
    endif
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -avoid-version \
                         -Wl,-exported_symbols_list -Wl,EnhancedDisassembly.exports \
                         -Wl,-dead_strip \
                         -Wl,-seg1addr -Wl,0xE0000000 

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions)  \
                            -no-undefined -Wl,-install_name \
                            -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

EDInfo.inc:	$(TBLGEN)
	$(Echo) "Building semantic information header"
	$(Verb) $(TableGen) -o $(call SYSPATH, $@) -gen-enhanced-disassembly-header /dev/null

EnhancedDisassembly.exports:	EnhancedDisassembly.exports.noblocks EnhancedDisassembly.exports.blocks
	$(Echo) "Copying exports file"
ifeq ($(HOST_OS),Darwin)
	$(Verb) if [ $(DARWIN_VERS) -ge 10 ] ; \
	  then $(CP) -f $(PROJ_SRC_DIR)/EnhancedDisassembly.exports.blocks EnhancedDisassembly.exports ; \
	  else $(CP) -f $(PROJ_SRC_DIR)/EnhancedDisassembly.exports.noblocks EnhancedDisassembly.exports ; \
	fi
else
	$(Verb) $(CP) -f $(PROJ_SRC_DIR)/EnhancedDisassembly.exports.noblocks EnhancedDisassembly.exports
endif

clean::
	-$(Verb) $(RM) -f EDInfo.inc
	-$(Verb) $(RM) -f EnhancedDisassembly.exports
