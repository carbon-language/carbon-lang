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
LIBRARYNAME = crtend
BYTECODE_DESTINATION = $(CFERuntimeLibDir)

MainSrc      := crtend.c
GenericEHSrc := Exception.cpp
SJLJEHSrc    := SJLJ-Exception.cpp

EXTRA_DIST   := $(MainSrc) $(GenericEHSrc) $(SJLJEHSrc) \
                comp_main.lst comp_genericeh.lst comp_sjljeh.lst

include $(LEVEL)/Makefile.common

MainObj      := $(ObjDir)/crtend.bc
GenericEHObj := $(ObjDir)/Exception.bc
SJLJEHObj    := $(ObjDir)/SJLJ-Exception.bc

# __main and ctor/dtor support component
$(ObjDir)/comp_main.bc: $(MainObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(PROJ_SRC_DIR)/comp_main.lst \
	$(MainObj) -o $@

# Generic exception handling support runtime.
$(ObjDir)/comp_genericeh.bc: $(GenericEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(PROJ_SRC_DIR)/comp_genericeh.lst \
	$(GenericEHObj) -o $@ 

# setjmp/longjmp exception handling support runtime.
$(ObjDir)/comp_sjljeh.bc: $(SJLJEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(PROJ_SRC_DIR)/comp_sjljeh.lst \
	$(SJLJEHObj) -o $@

SYMBOLHACKEDOBJS := $(ObjDir)/comp_main.bc $(ObjDir)/comp_genericeh.bc \
                    $(ObjDir)/comp_sjljeh.bc

all-local:: $(LibName.BCA)

ifdef BYTECODE_DESTINATION
BytecodeDestDir := $(BYTECODE_DESTINATION)
else
BytecodeDestDir := $(PROJ_libdir)
endif

DestBytecodeLib = $(BytecodeDestDir)/lib$(LIBRARYNAME).a
install-bytecode-local:: $(DestBytecodeLib)
install-local:: $(DestBytecodeLib)

$(LibName.BCA): $(SYMBOLHACKEDOBJS) $(LibDir)/.dir $(LLVMToolDir)/llvm-ar
	$(Echo) Building $(BuildMode) Bytecode Archive $(notdir $@)
	$(Verb) $(RM) -f $@
	$(Verb) $(LArchive) $@ $(SYMBOLHACKEDOBJS)

$(DestBytecodeLib): $(BytecodeDestDir) $(LibName.BCA) 
	$(Echo) Installing $(BuildMode) Bytecode Archive $(DestBytecodeLib)
	$(Verb) $(DataInstall) $(LibName.BCA) $(DestBytecodeLib)

uninstall-local::
	$(Echo) Uninstalling $(BuildMode) Bytecode Archive $(DestBytecodeLib)
	-$(Verb) $(RM) -f $(DestBytecodeLib)


