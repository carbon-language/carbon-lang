# makefile for libpng using gcc (generic, static library)
# Copyright (C) 2000 Cosmin Truta
# Copyright (C) 1995 Guy Eric Schalnat, Group 42, Inc.
# For conditions of distribution and use, see copyright notice in png.h

# Location of the zlib library and include files
ZLIBINC = ../zlib
ZLIBLIB = ../zlib

# Compiler, linker, lib and other tools
CC = gcc
LD = $(CC)
AR = ar rcs
RANLIB = ranlib
RM = rm -f

CDEBUG = -g -DPNG_DEBUG=5
LDDEBUG =
CRELEASE = -O2
LDRELEASE = -s
CFLAGS = -I$(ZLIBINC) -Wall $(CRELEASE)
LDFLAGS = -L. -L$(ZLIBLIB) -lpng -lz -lm $(LDRELEASE)

# File extensions
O=.o
A=.a
E=

# Variables
OBJS = png$(O) pngerror$(O) pngget$(O) pngmem$(O) pngpread$(O) \
	pngread$(O) pngrio$(O) pngrtran$(O) pngrutil$(O) pngset$(O) \
	pngtrans$(O) pngwio$(O) pngwrite$(O) pngwtran$(O) pngwutil$(O)

# Targets
all: libpng$(A) pngtest$(E)

libpng$(A): $(OBJS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

test: pngtest$(E)
	./pngtest$(E)

pngtest$(E): pngtest$(O) libpng$(A)
	$(LD) -o $@ pngtest$(O) $(LDFLAGS)

clean:
	$(RM) *$(O) libpng$(A) pngtest$(E) pngout.png

png$(O): png.h pngconf.h
pngerror$(O): png.h pngconf.h
pngget$(O): png.h pngconf.h
pngmem$(O): png.h pngconf.h
pngpread$(O): png.h pngconf.h
pngread$(O): png.h pngconf.h
pngrio$(O): png.h pngconf.h
pngrtran$(O): png.h pngconf.h
pngrutil$(O): png.h pngconf.h
pngset$(O): png.h pngconf.h
pngtest$(O): png.h pngconf.h
pngtrans$(O): png.h pngconf.h
pngwio$(O): png.h pngconf.h
pngwrite$(O): png.h pngconf.h
pngwtran$(O): png.h pngconf.h
pngwutil$(O): png.h pngconf.h

