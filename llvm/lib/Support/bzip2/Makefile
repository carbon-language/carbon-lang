
SHELL=/bin/sh

# To assist in cross-compiling
CC=gcc
AR=ar
RANLIB=ranlib
LDFLAGS=

# Suitably paranoid flags to avoid bugs in gcc-2.7
BIGFILES=-D_FILE_OFFSET_BITS=64
CFLAGS=-Wall -Winline -O2 -fomit-frame-pointer -fno-strength-reduce $(BIGFILES)

# Where you want it installed when you do 'make install'
PREFIX=/usr


OBJS= blocksort.o  \
      huffman.o    \
      crctable.o   \
      randtable.o  \
      compress.o   \
      decompress.o \
      bzlib.o

all: libbz2.a bzip2 bzip2recover test

bzip2: libbz2.a bzip2.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o bzip2 bzip2.o -L. -lbz2

bzip2recover: bzip2recover.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o bzip2recover bzip2recover.o

libbz2.a: $(OBJS)
	rm -f libbz2.a
	$(AR) cq libbz2.a $(OBJS)
	@if ( test -f $(RANLIB) -o -f /usr/bin/ranlib -o \
		-f /bin/ranlib -o -f /usr/ccs/bin/ranlib ) ; then \
		echo $(RANLIB) libbz2.a ; \
		$(RANLIB) libbz2.a ; \
	fi

check: test
test: bzip2
	@cat words1
	./bzip2 -1  < sample1.ref > sample1.rb2
	./bzip2 -2  < sample2.ref > sample2.rb2
	./bzip2 -3  < sample3.ref > sample3.rb2
	./bzip2 -d  < sample1.bz2 > sample1.tst
	./bzip2 -d  < sample2.bz2 > sample2.tst
	./bzip2 -ds < sample3.bz2 > sample3.tst
	cmp sample1.bz2 sample1.rb2 
	cmp sample2.bz2 sample2.rb2
	cmp sample3.bz2 sample3.rb2
	cmp sample1.tst sample1.ref
	cmp sample2.tst sample2.ref
	cmp sample3.tst sample3.ref
	@cat words3

install: bzip2 bzip2recover
	if ( test ! -d $(PREFIX)/bin ) ; then mkdir -p $(PREFIX)/bin ; fi
	if ( test ! -d $(PREFIX)/lib ) ; then mkdir -p $(PREFIX)/lib ; fi
	if ( test ! -d $(PREFIX)/man ) ; then mkdir -p $(PREFIX)/man ; fi
	if ( test ! -d $(PREFIX)/man/man1 ) ; then mkdir -p $(PREFIX)/man/man1 ; fi
	if ( test ! -d $(PREFIX)/include ) ; then mkdir -p $(PREFIX)/include ; fi
	cp -f bzip2 $(PREFIX)/bin/bzip2
	cp -f bzip2 $(PREFIX)/bin/bunzip2
	cp -f bzip2 $(PREFIX)/bin/bzcat
	cp -f bzip2recover $(PREFIX)/bin/bzip2recover
	chmod a+x $(PREFIX)/bin/bzip2
	chmod a+x $(PREFIX)/bin/bunzip2
	chmod a+x $(PREFIX)/bin/bzcat
	chmod a+x $(PREFIX)/bin/bzip2recover
	cp -f bzip2.1 $(PREFIX)/man/man1
	chmod a+r $(PREFIX)/man/man1/bzip2.1
	cp -f bzlib.h $(PREFIX)/include
	chmod a+r $(PREFIX)/include/bzlib.h
	cp -f libbz2.a $(PREFIX)/lib
	chmod a+r $(PREFIX)/lib/libbz2.a
	cp -f bzgrep $(PREFIX)/bin/bzgrep
	ln $(PREFIX)/bin/bzgrep $(PREFIX)/bin/bzegrep
	ln $(PREFIX)/bin/bzgrep $(PREFIX)/bin/bzfgrep
	chmod a+x $(PREFIX)/bin/bzgrep
	cp -f bzmore $(PREFIX)/bin/bzmore
	ln $(PREFIX)/bin/bzmore $(PREFIX)/bin/bzless
	chmod a+x $(PREFIX)/bin/bzmore
	cp -f bzdiff $(PREFIX)/bin/bzdiff
	ln $(PREFIX)/bin/bzdiff $(PREFIX)/bin/bzcmp
	chmod a+x $(PREFIX)/bin/bzdiff
	cp -f bzgrep.1 bzmore.1 bzdiff.1 $(PREFIX)/man/man1
	chmod a+r $(PREFIX)/man/man1/bzgrep.1
	chmod a+r $(PREFIX)/man/man1/bzmore.1
	chmod a+r $(PREFIX)/man/man1/bzdiff.1
	echo ".so man1/bzgrep.1" > $(PREFIX)/man/man1/bzegrep.1
	echo ".so man1/bzgrep.1" > $(PREFIX)/man/man1/bzfgrep.1
	echo ".so man1/bzmore.1" > $(PREFIX)/man/man1/bzless.1
	echo ".so man1/bzdiff.1" > $(PREFIX)/man/man1/bzcmp.1

distclean: clean
clean: 
	rm -f *.o libbz2.a bzip2 bzip2recover \
	sample1.rb2 sample2.rb2 sample3.rb2 \
	sample1.tst sample2.tst sample3.tst

blocksort.o: blocksort.c
	@cat words0
	$(CC) $(CFLAGS) -c blocksort.c
huffman.o: huffman.c
	$(CC) $(CFLAGS) -c huffman.c
crctable.o: crctable.c
	$(CC) $(CFLAGS) -c crctable.c
randtable.o: randtable.c
	$(CC) $(CFLAGS) -c randtable.c
compress.o: compress.c
	$(CC) $(CFLAGS) -c compress.c
decompress.o: decompress.c
	$(CC) $(CFLAGS) -c decompress.c
bzlib.o: bzlib.c
	$(CC) $(CFLAGS) -c bzlib.c
bzip2.o: bzip2.c
	$(CC) $(CFLAGS) -c bzip2.c
bzip2recover.o: bzip2recover.c
	$(CC) $(CFLAGS) -c bzip2recover.c

DISTNAME=bzip2-1.0.2
tarfile:
	rm -f $(DISTNAME)
	ln -sf . $(DISTNAME)
	tar cvf $(DISTNAME).tar \
	   $(DISTNAME)/blocksort.c \
	   $(DISTNAME)/huffman.c \
	   $(DISTNAME)/crctable.c \
	   $(DISTNAME)/randtable.c \
	   $(DISTNAME)/compress.c \
	   $(DISTNAME)/decompress.c \
	   $(DISTNAME)/bzlib.c \
	   $(DISTNAME)/bzip2.c \
	   $(DISTNAME)/bzip2recover.c \
	   $(DISTNAME)/bzlib.h \
	   $(DISTNAME)/bzlib_private.h \
	   $(DISTNAME)/Makefile \
	   $(DISTNAME)/manual.texi \
	   $(DISTNAME)/manual.ps \
	   $(DISTNAME)/manual.pdf \
	   $(DISTNAME)/LICENSE \
	   $(DISTNAME)/bzip2.1 \
	   $(DISTNAME)/bzip2.1.preformatted \
	   $(DISTNAME)/bzip2.txt \
	   $(DISTNAME)/words0 \
	   $(DISTNAME)/words1 \
	   $(DISTNAME)/words2 \
	   $(DISTNAME)/words3 \
	   $(DISTNAME)/sample1.ref \
	   $(DISTNAME)/sample2.ref \
	   $(DISTNAME)/sample3.ref \
	   $(DISTNAME)/sample1.bz2 \
	   $(DISTNAME)/sample2.bz2 \
	   $(DISTNAME)/sample3.bz2 \
	   $(DISTNAME)/dlltest.c \
	   $(DISTNAME)/*.html \
	   $(DISTNAME)/README \
	   $(DISTNAME)/README.COMPILATION.PROBLEMS \
	   $(DISTNAME)/CHANGES \
	   $(DISTNAME)/libbz2.def \
	   $(DISTNAME)/libbz2.dsp \
	   $(DISTNAME)/dlltest.dsp \
	   $(DISTNAME)/makefile.msc \
	   $(DISTNAME)/Y2K_INFO \
	   $(DISTNAME)/unzcrash.c \
	   $(DISTNAME)/spewG.c \
	   $(DISTNAME)/mk251.c \
	   $(DISTNAME)/bzdiff \
	   $(DISTNAME)/bzdiff.1 \
	   $(DISTNAME)/bzmore \
	   $(DISTNAME)/bzmore.1 \
	   $(DISTNAME)/bzgrep \
	   $(DISTNAME)/bzgrep.1 \
	   $(DISTNAME)/Makefile-libbz2_so
	gzip -v $(DISTNAME).tar

# For rebuilding the manual from sources on my RedHat 7.2 box
manual: manual.ps manual.pdf manual.html

manual.ps: manual.texi
	tex manual.texi
	dvips -o manual.ps manual.dvi

manual.pdf: manual.ps
	ps2pdf manual.ps

manual.html: manual.texi
	texi2html -split_chapter manual.texi
