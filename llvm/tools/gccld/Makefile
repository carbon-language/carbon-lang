##===- tools/gccld/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

TOOLNAME = gccld
LINK_COMPONENTS = bcreader bcwriter ipo scalaropts ipa linker
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

ifdef DONT_USE_THIS
install-local:: $(PROJ_bindir)/gccld
	
$(PROJ_bindir)/gccld : gccld.sh Makefile
	$(Echo)Installing gccld shell script.
	$(Verb)sed "s#@TOOLDIR@#$(PROJ_bindir)#" $< > $@
	$(Verb)chmod 0755 $@

all-local:: $(ToolDir)/gccld
	
$(ToolDir)/gccld : gccld.sh Makefile
	$(Echo)Making $(ToolDir)/gccld shell script.
	$(Verb)sed "s#@TOOLDIR@#$(ToolDir)#" $< > $@
	$(Verb)chmod 0755 $@

clean-local::
	$(Verb)$(RM) -f $(ToolDir)/gccld
endif
