LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: analyze
clean ::
	rm -f opt

analyze : $(ObjectsG) Debug/.dir Depend/.dir
	$(LinkG) -o $@ $(ObjectsG) -lopt -lasmparser \
				   -lbcreader -lvmcore -lasmwriter -lanalysis
