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
# Since this archive has strange requirements, we use some custom rules for 
# building it.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
DONT_BUILD_RELINKED = 1
BYTECODE_LIBRARY = 1
LIBRARYNAME = crtend

# Note: We're fooling Makefile.rules here. The sources listed don't exist but
# it doesn't matter. Makefile.rules just uses $(SOURCES) to generate the set of
# object files to put in the library. Since we've specified BYTECODE_LIBRARY
# above, it changes the .c suffix to .bc suffix. Below are rules to build 
# each of those bytecode objects from the sources we do have in this directory.
SOURCES = comp_main.c comp_genericeh.c comp_sjljeh.c

EXTRA_DIST   := comp_main.lst comp_genericeh.lst comp_sjljeh.lst

include $(LEVEL)/Makefile.common

MainSrc      := crtend.c listend.ll
GenericEHSrc := Exception.cpp
SJLJEHSrc    := SJLJ-Exception.cpp

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
