LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: opt
clean ::
	rm -f opt

opt : $(ObjectsG) ../../lib/Optimizations/Debug/libopt.a
	$(LinkG) -o $@ $(ObjectsG) -lopt -lbcreader -lbcwriter \
                               -lasmwriter -lanalysis -lvmcore -lsupport
