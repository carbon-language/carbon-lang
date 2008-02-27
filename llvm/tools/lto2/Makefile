##===- tools/lto2/Makefile ---------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LTO

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before includeing Makefile.rules
include $(LEVEL)/Makefile.config

LINK_LIBS_IN_SHARED = 1
ifeq ($(OS),Darwin)
  SHARED_LIBRARY = 1
  DONT_BUILD_RELINKED = 1
else
  BUILD_ARCHIVE = 1
endif

LINK_COMPONENTS := $(TARGETS_TO_BUILD) ipo scalaropts linker bitreader bitwriter

include $(LEVEL)/Makefile.common

ifeq ($(OS),Darwin)
    # set dylib internal version number to llvmCore submission number
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
						-Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION)
    endif
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
				-no-undefined  -avoid-version \
                -Wl,-exported_symbols_list -Wl,$(PROJ_SRC_DIR)/lto.exports \
                -Wl,-dead_strip \
                -Wl,-install_name \
				-Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)" \
                -Wl,-seg1addr -Wl,0xE0000000 

endif
