LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: as
clean::
	rm -f as

as : $(ObjectsG)
	$(LinkG) -o as $(ObjectsG) -lasmparser -lbcwriter -lasmwriter -lanalysis -lvmcore  -lsupport
