##===- tools/llee/Makefile ---------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = LLVMexecve
SHARED_LIBRARY = 1
DONT_BUILD_RELINKED = 1
include $(LEVEL)/Makefile.common

all:: llee

llee: $(LLVMToolDir)/llee

$(LLVMToolDir)/llee: Makefile
	$(Echo) Constructing llee shell script
	$(Verb) echo exec env LD_PRELOAD=$(LibDir)/libLLVMexecve$(SHLIBEXT) $$\* > $@
	$(Verb) chmod u+x $@

clean::
	$(Verb) rm -f $(LLVMToolDir)/llee
