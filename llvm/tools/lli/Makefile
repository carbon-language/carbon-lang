LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: lli
clean::	
	rm -f lli

lli : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) \
		-lopt -lbcreader -lbcwriter \
		-lvmcore -lasmwriter -lanalysis -lsupport -ltarget
