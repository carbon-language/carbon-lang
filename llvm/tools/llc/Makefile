LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: llc
clean::	
	rm -f llc

llc : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) \
		-lselect -lsparc -ltarget \
		-lopt -lbcreader -lbcwriter \
		-lvmcore -lasmwriter -lanalysis -lsupport
