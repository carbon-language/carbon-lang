LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: as
clean::
	rm -f as

as : $(ObjectsG)
	$(LinkG) -o as $(ObjectsG) -lvmcore -lasmparser -lbcwriter -lanalysis -lasmwriter
