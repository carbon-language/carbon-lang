LEVEL = ../..
TOOLNAME = opt

USEDLIBS = bcreader bcwriter instrument profpaths \
	   sparc target.a mapping regalloc.a sched select codegen preopts \
           postopts.a \
	   livevar scalaropts \
	   ipo ipa.a datastructure transforms target.a analysis \
           transformutils vmcore support
TOOLLINKOPTS = -ldl $(ExportSymbols)

KEEP_SYMBOLS = 1

include $(LEVEL)/Makefile.common

