LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: opt
clean ::
	rm -f opt

opt : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) -lvmcore -lanalysis -lbcreader -lbcwriter \
                               -lopt -lasmwriter
