##===- tools/llee/Makefile ---------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = execve
SHARED_LIBRARY = 1
include $(LEVEL)/Makefile.common

all:: llee

llee: $(TOOLDIR)/llee

$(TOOLDIR)/llee: Makefile
	echo exec env LD_PRELOAD=$(TOOLDIR)/libexecve$(SHLIBEXT) $$\* > $@
	chmod u+x $@

clean::
	rm -f $(TOOLDIR)/llee
