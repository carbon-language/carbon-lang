##
# libcpp Makefile
##

SRCDIRS = .
DESTDIR = $(DSTROOT)

OBJROOT=.
SYMROOT=.
export TRIPLE=-apple-

ifeq (,$(RC_INDIGO))
	INSTALL_PREFIX=""
else
	INSTALL_PREFIX="$(SDKROOT)"
endif
INSTALL_DIR=$(DSTROOT)/$(INSTALL_PREFIX)

.PHONY: help installsrc clean installheaders do-installhdrs install

help::
	@echo "Use make install DSTROOT=<destination>"

installsrc:: $(SRCROOT)

	ditto $(SRCDIRS)/include $(SRCROOT)/include
	ditto $(SRCDIRS)/lib $(SRCROOT)/lib
	ditto $(SRCDIRS)/src $(SRCROOT)/src
	ditto $(SRCDIRS)/Makefile $(SRCROOT)/Makefile

clean::

installheaders:: do-installhdrs

install::

	cd lib && ./buildit
	ditto lib/libc++.1.dylib $(SYMROOT)/usr/lib/libc++.1.dylib
	cd lib && dsymutil -o $(SYMROOT)/libc++.1.dylib.dSYM \
	  $(SYMROOT)/usr/lib/libc++.1.dylib
	mkdir -p $(INSTALL_DIR)/usr/lib
	strip -S -o $(INSTALL_DIR)/usr/lib/libc++.1.dylib \
	  $(SYMROOT)/usr/lib/libc++.1.dylib
	cd $(INSTALL_DIR)/usr/lib && ln -s libc++.1.dylib libc++.dylib
