LEVEL = ../..
TOOLNAME = lli
USEDLIBS = bcreader bcwriter vmcore asmwriter analysis support \
	   target transforms
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common

all :: $(LEVEL)/tools/Debug/RuntimeLib.bc

Debug/RuntimeLib.c: RuntimeLib.lc
	cp -f $< $@

Debug/RuntimeLib.ll: Debug/RuntimeLib.c
	@-rm $@
	/home/vadve/lattner/cvs/gcc_install/bin/gcc $< -S -o $@

$(LEVEL)/tools/Debug/RuntimeLib.bc: Debug/RuntimeLib.ll
	LD_LIBRARY_PATH=../lib/Debug:/usr/dcs/software/evaluation/encap/gcc-3.0.4/lib/ ../Debug/gccas $< -o $@


