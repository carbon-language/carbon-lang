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

configure: autoconf/configure.ac
	cd autoconf && $(AUTOCONF) -o ../configure configure.ac

