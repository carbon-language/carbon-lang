LEVEL = ../../..
LIBRARYNAME = sparc

ExtraSource = Debug/Sparc.burm.cpp 

include $(LEVEL)/Makefile.common

ifdef ENABLE_OPTIMIZED
  DEBUG_FLAG = 
else
  DEBUG_FLAG = -D_DEBUG
endif

Debug/Sparc.burm.cpp: Debug/Sparc.burm Debug/.dir
	$(RunBurg) $< -o $@

$(BUILD_ROOT)/Debug/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_ROOT)/Release/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_ROOT)/Profile/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileP) $< -o $@

Debug/Sparc.burg.in1 : Sparc.burg.in Debug/.dir
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

Debug/Sparc.burm : Debug/Sparc.burg.in1
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

$(BUILD_ROOT)/Depend/Sparc.burm.d: $(BUILD_ROOT)/Depend/.dir
	touch $@

SparcV9CodeEmitter.cpp: SparcV9CodeEmitter.inc

TEMP_EMITTER_INC = _temp_emitter.inc

SparcV9CodeEmitter.inc: SparcV9.td SparcV9_F2.td SparcV9_F3.td SparcV9_F4.td SparcV9_Reg.td
	@echo "TableGen-erating $@"
	cpp -P SparcV9.td | $(TBLGEN) -gen-emitter > $(TEMP_EMITTER_INC) 
	mv -f $(TEMP_EMITTER_INC) SparcV9CodeEmitter.inc

clean::
	rm -f SparcV9CodeEmitter.inc
