LEVEL = ../../..
LIBRARYNAME = sparc
ExtraSource = Debug/Sparc.burm.cpp 

include $(LEVEL)/Makefile.common

Debug/Sparc.burm.cpp: $(BUILD_ROOT)/Debug/Sparc.burm Debug/.dir
	$(RunBurg) $< -o $@

$(BUILD_ROOT)/Debug/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_ROOT)/Debug/Sparc.burg.in1 : Sparc.burg.in $(BUILD_ROOT)/Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

$(BUILD_ROOT)/Debug/Sparc.burm : $(BUILD_ROOT)/Debug/Sparc.burg.in1 $(BUILD_ROOT)/Debug/.dir
	g++ -E -I$(LEVEL)/include -D_DEBUG -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

$(BUILD_ROOT)/Depend/Sparc.burm.d: $(BUILD_ROOT)/Depend/.dir
	touch $@

