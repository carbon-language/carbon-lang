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
Source       := $(MainSrc) $(GenericEHSrc) $(SJLJEHSrc) $(CXXEHSrc)

include $(LEVEL)/Makefile.common

# CRTEND_A - The result of making 'all' - the final archive file.
CRTEND_A   = $(LIBDIR)/libcrtend.a
all:: $(CRTEND_A)

# Installation simply requires copying the archive to it's new home.
$(DESTDIR)$(bytecode_libdir)/libcrtend.a: $(CRTEND_A)
	$(INSTALL) $(CRTEND_A) $(DESTDIR)$(bytecode_libdir)

install:: $(DESTDIR)$(bytecode_libdir)/libcrtend.a
install-bytecode:: $(DESTDIR)$(bytecode_libdir)/libcrtend.a


# The four components described in the README
Components := main genericeh sjljeh
ComponentLibs := $(Components:%=$(OBJDIR)/comp_%.bc)


# We build libcrtend.a from the four components described in the README.
$(CRTEND_A) : $(ComponentLibs) $(LIBDIR)/.dir
	@echo Building final libcrtend.a file from components
	$(VERB) $(Archive) $@ $(ComponentLibs)

MainObj      := $(OBJDIR)/crtend.bc $(OBJDIR)/listend.bc
GenericEHObj := $(OBJDIR)/Exception.bc
SJLJEHObj    := $(OBJDIR)/SJLJ-Exception.bc

# __main and ctor/dtor support component
$(OBJDIR)/comp_main.bc: $(MainObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(GCCLD) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_main.lst $(MainObj) -o $@

# Generic exception handling support runtime.
$(OBJDIR)/comp_genericeh.bc: $(GenericEHObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(GCCLD) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_genericeh.lst $(GenericEHObj) -o $@

# setjmp/longjmp exception handling support runtime.
$(OBJDIR)/comp_sjljeh.bc: $(SJLJEHObj)
	@echo Linking $(notdir $@) component...
	$(VERB) $(GCCLD) -link-as-library -internalize-public-api-file=$(BUILD_SRC_DIR)/comp_sjljeh.lst $(SJLJEHObj) -o $@


