LEVEL = ../../..

include $(LEVEL)/Makefile.common

LIB32 = ../Output/libinstr32.a
LIB64 = ../Output/libinstr64.a 
all:: $(LIB32) $(LIB64)

tracelib: tracelib.c
	g++ -g -DTEST_INSTRLIB $< -o $@

Debug/tracelib32.o: tracelib.c Debug/.dir
	g++ -c -g $< -o $@

Debug/tracelib64.o: tracelib.c Debug/.dir
	cc -c -xarch=v9 -g $< -o $@

$(LIB32): Debug/tracelib32.o ../Output/.dir
	ar r $@ $<

$(LIB64): Debug/tracelib64.o ../Output/.dir
	ar r $@ $<

test: tracelib

tracelib.c: tracelib.h
