##===- lib/CompilerDriver/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..

# We don't want this library to appear in `llvm-config --libs` output, so its
# name doesn't start with "LLVM".

LIBRARYNAME = libCompilerDriver
LLVMLIBS = LLVMSupport.a LLVMSystem.a
LOADABLE_MODULE := 1
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

FullLibName = $(LIBRARYNAME)$(SHLIBEXT)

all-local:: $(ToolDir)/$(FullLibName)

# Copy the library to the bin dir so that llvmc can find it.
$(ToolDir)/$(FullLibName): $(LibDir)/$(FullLibName) $(ToolDir)/.dir
	$(Echo) Copying $(BuildMode) Shared Library $(FullLibName) to $@
	-$(Verb) $(CP) $< $@

clean-local::
	$(Echo) Removing $(BuildMode) Shared Library $(FullLibName) \
	from $(ToolDir)
	-$(Verb) $(RM) -f $(ToolDir)/$(FullLibName)
