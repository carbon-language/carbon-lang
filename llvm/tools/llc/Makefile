LEVEL = ../..

DIRS  = 

LIBRARYNAME = llc

## List source files in link order
Source  = \
	  llc.o \
	  LLCOptions.o

include $(LEVEL)/Makefile.common

all:: llc

clean::	
	rm -f llc

llc : $(ObjectsG)
	$(LinkG) -o $@ -static \
		-lllc -lselect -lsparc -ltarget \
		-lopt -lbcreader -lbcwriter \
		-lvmcore -lasmwriter -lanalysis -lsupport
