# makefile for libpng, MACOS X
# Copyright (C) 2002 Glenn Randers-Pehrson
# Copyright (C) 1995 Guy Eric Schalnat, Group 42, Inc.
# Modified by Karin Kosina <kyrah@sim.no> 20011010:
# build shared library (*.dylib)
# For conditions of distribution and use, see copyright notice in png.h

# where make install puts libpng.a and png.h
prefix=/usr/local

# Where the zlib library and include files are located
#ZLIBLIB=/usr/local/lib
#ZLIBINC=/usr/local/include
ZLIBLIB=../zlib
ZLIBINC=../zlib

CC=cc

PNGMAJ = 0
PNGMIN = 1.2.5
PNGVER = $(PNGMAJ).$(PNGMIN)

CFLAGS=-fno-common -I$(ZLIBINC) -O # -g -DPNG_DEBUG=5
LDFLAGS=-L. -L$(ZLIBLIB) -lpng -lz -current_version $(PNGVER)

LIBNAME=libpng12
SHAREDLIB_POSTFIX=dylib
INCPATH=$(prefix)/include
LIBPATH=$(prefix)/lib
MANPATH=$(prefix)/man
BINPATH=$(prefix)/bin

# override DESTDIR= on the make install command line to easily support
# installing into a temporary location.  Example:
#
#    make install DESTDIR=/tmp/build/libpng
#
# If you're going to install into a temporary location
# via DESTDIR, $(DESTDIR)$(prefix) must already exist before
# you execute make install.
DESTDIR=

DB=$(DESTDIR)$(BINPATH)
DI=$(DESTDIR)$(INCPATH)
DL=$(DESTDIR)$(LIBPATH)
DM=$(DESTDIR)$(MANPATH)

#RANLIB=echo
RANLIB=ranlib

OBJS = png.o pngset.o pngget.o pngrutil.o pngtrans.o pngwutil.o \
	pngread.o pngrio.o pngwio.o pngwrite.o pngrtran.o \
	pngwtran.o pngmem.o pngerror.o pngpread.o

all: libpng.a pngtest shared libpng.pc libpng-config

shared: $(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX)

libpng.pc:
	cat scripts/libpng.pc.in | sed -e s\!@PREFIX@!$(prefix)! | \
	   sed -e s/-lm// > libpng.pc

libpng-config:
	( cat scripts/libpng-config-head.in; \
	echo prefix=\"$(prefix)\"; \
	echo I_opts=\"-I$(INCPATH)/$(LIBNAME)\"; \
	echo L_opts=\"-L$(LIBPATH)\"; \
	echo libs=\"-lpng12 -lz\"; \
	cat scripts/libpng-config-body.in ) > libpng-config
	chmod +x libpng-config

libpng.a: $(OBJS)
	ar rc $@  $(OBJS)
	$(RANLIB) $@

$(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX): $(OBJS)
	cc -dynamiclib -flat_namespace -undefined suppress -o $@ $(OBJS)

libpng.3.$(PNGMIN).$(SHAREDLIB_POSTFIX): $(OBJS)
	cc -dynamiclib -compatibility_version 3 -flat_namespace \
	-undefined suppress -o $@ $(OBJS)

pngtest: pngtest.o libpng.a
	$(CC) -o pngtest $(CFLAGS) pngtest.o $(LDFLAGS)

test: pngtest
	./pngtest

install-headers: png.h pngconf.h
	-@if [ ! -d $(DI) ]; then mkdir $(DI); fi
	-@if [ ! -d $(DI)/$(LIBNAME) ]; then mkdir $(DI)/$(LIBNAME); fi
	cp png.h pngconf.h $(DI)/$(LIBNAME)
	chmod 644 $(DI)/$(LIBNAME)/png.h $(DI)/$(LIBNAME)/pngconf.h
	-@/bin/rm -f $(DI)/png.h $(DI)/pngconf.h
	-@/bin/rm -f $(DI)/libpng
	(cd $(DI); ln -f -s $(LIBNAME) libpng; ln -f -s $(LIBNAME)/* .)

install-static: install-headers libpng.a
	-@if [ ! -d $(DL) ]; then mkdir $(DL); fi
	cp libpng.a $(DL)/$(LIBNAME).a
	chmod 644 $(DL)/$(LIBNAME).a
	-@/bin/rm -f $(DL)/libpng.a
	(cd $(DL); ln -f -s $(LIBNAME).a libpng.a)

install-shared: install-headers $(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX) \
	libpng.pc libpng.3.$(PNGMIN).$(SHAREDLIB_POSTFIX)
	-@if [ ! -d $(DL) ]; then mkdir $(DL); fi
	-@/bin/rm -f \
	$(DL)/$(LIBNAME).$(PNGVER)*.$(SHAREDLIB_POSTFIX)
	-@/bin/rm -f $(DL)/$(LIBNAME).$(SHAREDLIB_POSTFIX)
	-@/bin/rm -f libpng.$(SHARED_POSTFIX)
	-@/bin/rm -f libpng.3.$(SHARED_POSTFIX)
	-@/bin/rm -f libpng.3.$(PNGMIN)*.$(SHARED_POSTFIX)
	cp libpng.3.$(PNGMIN).$(SHAREDLIB_POSTFIX) $(DL)
	cp $(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX) $(DL)
	chmod 755 $(DL)/$(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX)
	chmod 755 $(DL)/libpng.3.$(PNGMIN).$(SHAREDLIB_POSTFIX)
	(cd $(DL); \
	ln -f -s libpng.3.$(PNGMIN).$(SHARED_POSTFIX) \
	libpng.3.$(SHARED_POSTFIX); \
	ln -f -s libpng.3.$(SHARED_POSTFIX) libpng.$(SHARED_POSTFIX); \
	ln -f -s $(LIBNAME).$(PNGVER).$(SHARED_POSTFIX) \
	libpng.$(SHARED_POSTFIX); \
	ln -f -s libpng.3.$(PNGMIN).$(SHARED_POSTFIX) \
	libpng.3.$(SHARED_POSTFIX); \
	ln -f -s $(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX) \
	$(LIBNAME).$(PNGMAJ).$(SHAREDLIB_POSTFIX); \
	ln -f -s $(LIBNAME).$(PNGMAJ).$(SHAREDLIB_POSTFIX) \
	$(LIBNAME).$(SHAREDLIB_POSTFIX))
	-@if [ ! -d $(DL)/pkgconfig ]; then mkdir $(DL)/pkgconfig; fi
	-@/bin/rm -f $(DL)/pkgconfig/$(LIBNAME).pc
	-@/bin/rm -f $(DL)/pkgconfig/libpng.pc
	cp libpng.pc $(DL)/pkgconfig/$(LIBNAME).pc
	chmod 644 $(DL)/pkgconfig/$(LIBNAME).pc
	(cd $(DL)/pkgconfig; ln -f -s $(LIBNAME).pc libpng.pc)

install-man: libpng.3 libpngpf.3 png.5
	-@if [ ! -d $(DM) ]; then mkdir $(DM); fi
	-@if [ ! -d $(DM)/man3 ]; then mkdir $(DM)/man3; fi
	-@/bin/rm -f $(DM)/man3/libpng.3
	-@/bin/rm -f $(DM)/man3/libpngpf.3
	cp libpng.3 $(DM)/man3
	cp libpngpf.3 $(DM)/man3
	-@if [ ! -d $(DM)/man5 ]; then mkdir $(DM)/man5; fi
	-@/bin/rm -f $(DM)/man5/png.5
	cp png.5 $(DM)/man5

install-config: libpng-config
	-@if [ ! -d $(DB) ]; then mkdir $(DB); fi
	-@/bin/rm -f $(DB)/libpng-config
	-@/bin/rm -f $(DB)/$(LIBNAME)-config
	cp libpng-config $(DB)/$(LIBNAME)-config
	chmod 755 $(DB)/$(LIBNAME)-config
	(cd $(DB); ln -sf $(LIBNAME)-config libpng-config)

install: install-static install-shared install-man install-config

# If you installed in $(DESTDIR), test-installed won't work until you
# move the library to its final location.

test-installed:
	$(CC) $(CFLAGS) \
	   `$(BINPATH)/libpng12-config --cflags` pngtest.c \
	   -L$(ZLIBLIB) \
	   -o pngtesti `$(BINPATH)/libpng12-config --ldflags`
	./pngtesti pngtest.png

clean:
	rm -f *.o libpng.a pngtest pngout.png libpng-config \
	$(LIBNAME).$(PNGVER).$(SHAREDLIB_POSTFIX) \
	$(LIBNAME).$(SHAREDLIB_POSTFIX) \
	libpng.3.$(PNGMIN).$(SHAREDLIB_POSTFIX) \
	libpng.pc

DOCS = ANNOUNCE CHANGES INSTALL KNOWNBUG LICENSE README TODO Y2KINFO
writelock:
	chmod a-w *.[ch35] $(DOCS) scripts/*

# DO NOT DELETE THIS LINE -- make depend depends on it.

png.o: png.h pngconf.h
pngerror.o: png.h pngconf.h
pngrio.o: png.h pngconf.h
pngwio.o: png.h pngconf.h
pngmem.o: png.h pngconf.h
pngset.o: png.h pngconf.h
pngget.o: png.h pngconf.h
pngread.o: png.h pngconf.h
pngrtran.o: png.h pngconf.h
pngrutil.o: png.h pngconf.h
pngtest.o: png.h pngconf.h
pngtrans.o: png.h pngconf.h
pngwrite.o: png.h pngconf.h
pngwtran.o: png.h pngconf.h
pngwutil.o: png.h pngconf.h
pngpread.o: png.h pngconf.h

