LEVEL = ../..
TOOLNAME = lli
USEDLIBS = opt bcreader bcwriter vmcore asmwriter analysis support \
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
	../Debug/gccas $< -o $@


