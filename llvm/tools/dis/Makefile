LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: dis
clean ::
	rm -f dis

dis : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) -lbcreader -lasmwriter -lanalysis -lvmcore -lsupport

