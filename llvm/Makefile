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

AUTOCONF = autoconf
AUTOHEADER = autoheader

configure: autoconf/configure.ac autoconf/aclocal.m4
	cd autoconf && $(AUTOCONF) -o ../configure configure.ac

include/Config/config.h.in: autoconf/configure.ac autoconf/aclocal.m4
	autoheader -I autoconf autoconf/configure.ac

