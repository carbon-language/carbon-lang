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

EXTRA_DIST   := comp_main.lst comp_genericeh.lst comp_sjljeh.lst

MainSrc      := crtend.c listend.ll
GenericEHSrc := Exception.cpp
SJLJEHSrc    := SJLJ-Exception.cpp
Source       := $(MainSrc) $(GenericEHSrc) $(SJLJEHSrc) $(CXXEHSrc)

include $(LEVEL)/Makefile.common

# CRTEND_A - The result of making 'all' - the final archive file.
CRTEND_A   = $(LibDir)/libcrtend.a
all:: $(CRTEND_A)

# Installation simply requires copying the archive to it's new home.
$(bytecode_libdir)/libcrtend.a: $(CRTEND_A) $(bytecode_libdir)
	$(Echo) Installing $(CRTEND_A)
	$(Verb) $(INSTALL) $(CRTEND_A) $(bytecode_libdir)

install:: $(DESTDIR)$(bytecode_libdir)/libcrtend.a
install-bytecode:: $(DESTDIR)$(bytecode_libdir)/libcrtend.a

clean-local::
	$(Verb)$(RM) -f $(CRTEND_A)

# The three components described in the README
Components := main genericeh sjljeh
ComponentLibs := $(Components:%=$(ObjDir)/comp_%.bc)


# We build libcrtend.a from the four components described in the README.
$(CRTEND_A) : $(ComponentLibs) $(LibDir)/.dir
	$(Echo) Building final libcrtend.a file from components
	$(Verb) $(LArchive) $@ $(ComponentLibs)

MainObj      := $(ObjDir)/crtend.bc $(ObjDir)/listend.bc
GenericEHObj := $(ObjDir)/Exception.bc
SJLJEHObj    := $(ObjDir)/SJLJ-Exception.bc

# __main and ctor/dtor support component
$(ObjDir)/comp_main.bc: $(MainObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_main.lst \
	$(MainObj) -o $@ \

# Generic exception handling support runtime.
$(ObjDir)/comp_genericeh.bc: $(GenericEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_genericeh.lst \
	$(GenericEHObj) -o $@

# setjmp/longjmp exception handling support runtime.
$(ObjDir)/comp_sjljeh.bc: $(SJLJEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_sjljeh.lst \
	$(SJLJEHObj) -o $@
