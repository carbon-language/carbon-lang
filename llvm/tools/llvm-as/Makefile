LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: as
clean::
	rm -f as

as : $(ObjectsG) ../../lib/Assembly/Parser/Debug/libasmparser.a \
		 ../../lib/VMCore/Debug/libvmcore.a
	$(LinkG) -o as $(ObjectsG) -lasmparser -lbcwriter -lasmwriter -lanalysis -lvmcore  -lsupport
