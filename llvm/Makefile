##===- ./Makefile ------------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = .
DIRS = lib/Support utils lib tools runtime
OPTIONAL_DIRS = projects

include $(LEVEL)/Makefile.common

test :: all
	cd test; $(MAKE)

distclean:: clean
	$(VERB) $(RM) -rf $(LEVEL)/Makefile.config \
	                  $(LEVEL)/include/Config/config.h \
	                  $(LEVEL)/autoconf/autom4te.cache \
	                  $(LEVEL)/config.log \
	                  $(LEVEL)/TAGS

tools-only: DIRS = $(DIRS:runtime=)
tools-only: all

configure: autoconf/configure.ac autoconf/aclocal.m4
	cd autoconf && aclocal && autoconf -o ../configure configure.ac

include/Config/config.h.in: autoconf/configure.ac autoconf/aclocal.m4
	autoheader -I autoconf autoconf/configure.ac

# Install support for llvm include files:
.PHONY: install-includes

install-includes:
	$(MKDIR) $(includedir)/llvm
	cd include && find * '!' '(' -name '*~' -o -name .cvsignore ')' -print | grep -v CVS | pax -rwdvpe $(DESTDIR)$(includedir)/llvm

install:: install-includes

# Build tags database for Emacs/Xemacs:
.PHONY: tags

TAGS: tags

all:: tags

tags:
	$(ETAGS) $(ETAGSFLAGS) `find $(wildcard $(SourceDir)/include $(SourceDir)/lib $(SourceDir)/tools) -name '*.cpp' -o -name '*.h'`

