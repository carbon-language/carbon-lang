LEVEL = ../..

TOOLNAME = bugpoint

OPTLIBS  = instrument profpaths scalaropts ipo
ANALIBS  = datastructure ipa target.a analysis

USEDLIBS = ipo scalaropts $(ANALIBS) \
           transformutils asmparser bcreader bcwriter vmcore support

TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common
