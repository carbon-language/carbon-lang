LEVEL = ../../..
LIBRARYNAME = sparc

ExtraSource = Sparc.burm.cpp 

include $(LEVEL)/Makefile.common

Debug/Sparc.burg.in1 : Sparc.burg.in Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

Debug/Sparc.burg : Debug/Sparc.burg.in1 Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

Depend/Sparc.burm.d:
	touch $@

