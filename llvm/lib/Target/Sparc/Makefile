LEVEL = ../../..
LIBRARYNAME = sparc

## List source files in link order
Source  = \
	  Sparc.cpp \
	  Sparc.burm.cpp \
	  SparcInstrSelection.cpp \
	  SparcRegClassInfo.cpp \
	  SparcRegInfo.cpp EmitAssembly.cpp

include $(LEVEL)/Makefile.common

Debug/Sparc.burg.in1 : Sparc.burg.in
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

Debug/Sparc.burg : Debug/Sparc.burg.in1
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

