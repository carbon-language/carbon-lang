LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: analyze
clean ::
	rm -f analyze

LIBDEPS =  ../../lib/Optimizations/Debug/libopt.a  ../../lib/Analysis/Debug/libanalysis.a

analyze : $(ObjectsG) Debug/.dir Depend/.dir $(LIBDEPS)
	$(LinkG) -o $@ $(ObjectsG) -lopt -lasmparser \
				   -lbcreader -lvmcore -lasmwriter -lanalysis \
                                   -lopt -lsupport
