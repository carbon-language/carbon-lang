##===- tools/gccas/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

include $(LEVEL)/Makefile.common

install-local:: $(PROJ_bindir)/gccas
	
$(PROJ_bindir)/gccas : gccas.sh
	$(Echo)Installing gccas shell script.
	$(Verb)sed "s#@TOOLDIR@#$(PROJ_bindir)#" $< > $@

all-local:: $(ToolDir)/gccas
	
$(ToolDir)/gccas : gccas.sh
	$(Echo)Making $(ToolDir)/gccas shell script.
	$(Verb)sed "s#@TOOLDIR@#$(ToolDir)#" $< > $@

clean-local::
	$(Verb)$(RM) -f $(ToolDir)/gccas
