LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: opt
clean ::
	rm -f opt

opt : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) -lopt -lbcreader -lbcwriter \
                               -lasmwriter -lanalysis -lvmcore 
