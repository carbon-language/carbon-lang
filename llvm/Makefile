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

tools-only:
	@for dir in lib/Support utils lib tools; do $(MAKE) -C $$dir; done

AUTOCONF = autoconf
AUTOHEADER = autoheader

configure: autoconf/configure.ac autoconf/aclocal.m4
	cd autoconf && $(AUTOCONF) -o ../configure configure.ac

include/Config/config.h.in: autoconf/configure.ac autoconf/aclocal.m4
	$(AUTOHEADER) -I autoconf autoconf/configure.ac

