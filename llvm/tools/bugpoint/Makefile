LEVEL = ../..

TOOLNAME = bugpoint

OPTLIBS  = transforms instrument profpaths 
ANALIBS  = datastructure ipa target.a 

USEDLIBS = ipo scalaropts analysis $(OPTLIBS) $(ANALIBS) \
           transformutils asmparser bcreader bcwriter vmcore support

TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common
