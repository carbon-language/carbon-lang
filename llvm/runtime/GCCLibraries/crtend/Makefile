##===- runtime/GCCLibraries/crtend/Makefile ----------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This directory contains the C and C++ runtime libraries for the LLVM GCC
# front-ends.  See the README.txt file for more details.
#
# Since this archive has strange requirements, we use almost all custom rules
# for building it.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
DONT_BUILD_RELINKED=1

MainSrc      := crtend.c listend.ll
GenericEHSrc := Exception.cpp
SJLJEHSrc    := SJLJ-Exception.cpp
CXXEHSrc     := C++-Exception.cpp
Source       := $(MainSrc) $(GenericEHSrc) $(SJLJEHSrc) $(CXXEHSrc)

include $(LEVEL)/Makefile.common

# CRTEND_A - The result of making 'all' - the final archive file.
CRTEND_A   = $(DESTLIBBYTECODE)/crtend.a
all:: $(CRTEND_A)

# Installation simply requires copying the archive to it's new home.
$(LLVMGCCDIR)/bytecode-libs/crtend.a: $(CRTEND_A)
	cp $< $@

install:: $(LLVMGCCDIR)/bytecode-libs/crtend.a


# The four components described in the README
Components := main genericeh sjljeh cxxeh
ComponentLibs := $(Components:%=$(BUILD_OBJ_DIR)/BytecodeObj/comp_%.bc)


# We build crtend.a from the four components described in the README.
$(CRTEND_A) : $(ComponentLibs)
	@echo Building final crtend.a file from components
	$(VERB) $(AR) $@ $(ComponentLibs)

MainObj      := $(BUILD_OBJ_DIR)/BytecodeObj/crtend.bc \
                $(BUILD_OBJ_DIR)/BytecodeObj/listend.bc
GenericEHObj := $(BUILD_OBJ_DIR)/BytecodeObj/Exception.bc
SJLJEHObj    := $(BUILD_OBJ_DIR)/BytecodeObj/SJLJ-Exception.bc
CXXEHObj     := $(BUILD_OBJ_DIR)/BytecodeObj/C++-Exception.bc

# __main and ctor/dtor support component
$(BUILD_OBJ_DIR)/BytecodeObj/comp_main.bc: $(MainObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(LGCCLDPROG) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_main.lst $(MainObj) -o $@

# Generic exception handling support runtime.
$(BUILD_OBJ_DIR)/BytecodeObj/comp_genericeh.bc: $(GenericEHObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(LGCCLDPROG) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_genericeh.lst $(GenericEHObj) -o $@

# setjmp/longjmp exception handling support runtime.
$(BUILD_OBJ_DIR)/BytecodeObj/comp_sjljeh.bc: $(SJLJEHObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(LGCCLDPROG) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_sjljeh.lst $(SJLJEHObj) -o $@

# C++ exception handling support runtime.
$(BUILD_OBJ_DIR)/BytecodeObj/comp_cxxeh.bc: $(CXXEHObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(LGCCLDPROG) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_cxxeh.lst $(CXXEHObj) -o $@


