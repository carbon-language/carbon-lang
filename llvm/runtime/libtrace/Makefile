LEVEL = ../../..

LIBNAME = instr

include ../Makefile.libs

## We need the native libraries for libinstr because they are directly
## linked into the native CBE or LLC code.  That is the only practical
## way to debug them!

LIBINSTR = ../Output/libinstr.$(ARCH).a
all:: $(LIBINSTR)

tracelib: tracelib.c
	g++ -g -DTEST_INSTRLIB $< -o $@

Debug/tracelib.o: tracelib.c Debug/.dir
	$(CompileC) -g $< -o $@

$(LIBINSTR): Debug/tracelib.o ../Output/.dir
	ar r $@ $<

tracelib.c: tracelib.h
