LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: dis
clean ::
	rm -f dis

dis : $(ObjectsG)
	$(LinkG) -o $@ $(ObjectsG) -lvmcore -lasmwriter -lanalysis \
				               -lbcreader
