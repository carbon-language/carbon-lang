LEVEL = ../..
TOOLNAME = lli
USEDLIBS = opt bcreader bcwriter vmcore asmwriter analysis support \
	   target transforms
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common

all :: $(LEVEL)/tools/Debug/RuntimeLib.bc

Debug/RuntimeLib.c: RuntimeLib.lc
	cp -f $< $@

Debug/RuntimeLib.o: Debug/RuntimeLib.c
	/home/vadve/lattner/cvs/gcc_install/bin/gcc $< -c -o $@

$(LEVEL)/tools/Debug/RuntimeLib.bc: Debug/RuntimeLib.o
	../Debug/opt -dce $< -o $@ -f -q


