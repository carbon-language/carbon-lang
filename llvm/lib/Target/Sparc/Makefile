LEVEL = ../../..
LIBRARYNAME = sparc
ExtraSource = Debug/Sparc.burm.cpp 

#all:: Debug/Sparc.burg
#all:: Sparc.burm.cpp
include $(LEVEL)/Makefile.common

Debug/Sparc.burm.cpp: Debug/Sparc.burm
	$(RunBurg) $< -o $@

Debug/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

Debug/Sparc.burg.in1 : Sparc.burg.in Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

Debug/Sparc.burm : Debug/Sparc.burg.in1 Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

Depend/Sparc.burm.d:
	touch $@

