LEVEL = ../../../..

DIRS  = 

LIBRARYNAME = sparc

## List source files in link order
Source  = \
	  Sparc.o \
	  Sparc.burm.o \
	  SparcInstrSelection.o

include $(LEVEL)/Makefile.common

BURG = /home/vadve/vadve/Research/DynOpt/Burg/burg -I

%.burm.cpp: %.burg
	$(BURG) -o $@ $<

