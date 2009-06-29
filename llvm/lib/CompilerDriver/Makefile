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

# Copy the library to the bin dir so that llvmc can find it.
all-local::
	$(Echo) Copying $(BuildMode) Shared Library $(FullLibName) \
	to $(ToolDir)
	-$(Verb) $(CP) $(LibDir)/$(FullLibName) $(ToolDir)/

clean-local::
	$(Echo) Removing $(BuildMode) Shared Library $(FullLibName) \
	from $(ToolDir)
	-$(Verb) $(RM) -f $(ToolDir)/$(FullLibName)
