LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: llc
clean::	
	rm -f llc

llc : $(ObjectsG) $(LibsG)
	$(LinkG) -o $@ $(ObjectsG) \
		-lsched -lselect -lsparc -ltarget \
		-lopt -llivevar -lbcreader -lbcwriter \
		-lvmcore -lasmwriter -lanalysis -lsupport
