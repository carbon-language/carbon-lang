LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: analyze
clean ::
	rm -f opt

analyze : $(ObjectsG) Debug/.dir Depend/.dir
	$(LinkG) -o $@ $(ObjectsG) -lvmcore -lanalysis -lbcreader \
                               -lopt -lasmwriter -lasmparser
