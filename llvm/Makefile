#===- ./Makefile -------------------------------------------*- Makefile -*--===#
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#
LEVEL = .
DIRS = lib/System lib/Support utils lib tools 


ifneq ($(MAKECMDGOALS),tools-only)
DIRS += runtime
OPTIONAL_DIRS = examples projects
endif

EXTRA_DIST := llvm.spec include configure \
	      autoconf/AutoRegen.sh autoconf/LICENSE.TXT autoconf/README.TXT \
	      autoconf/aclocal.m4 autoconf/config.guess autoconf/config.sub \
	      autoconf/configure.ac autoconf/depcomp autoconf/install-sh \
	      autoconf/ltmain.sh autoconf/missing autoconf/mkinstalldirs \
	      autoconf/m4
include $(LEVEL)/Makefile.common

dist-hook::
	@$(ECHO) Eliminating CVS directories from distribution
	$(VERB) rm -rf `find $(TopDistDir) -type d -name CVS -print`

test :: all
	cd test; $(MAKE)

tools-only: all
