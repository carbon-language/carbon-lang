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

$(BUILD_OBJ_DIR)/Debug/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_OBJ_DIR)/Release/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_OBJ_DIR)/Profile/Sparc.burm.o: Debug/Sparc.burm.cpp
	$(CompileP) $< -o $@

Debug/Sparc.burg.in1 : Sparc.burg.in Debug/.dir
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | sed '/^# /d' | sed 's/Ydefine/#define/' > $@

Debug/Sparc.burm : Debug/Sparc.burg.in1
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | sed '/^# /d' | sed 's/Xinclude/#include/g' | sed 's/Xdefine/#define/g' > $@

$(BUILD_OBJ_DIR)/Depend/Sparc.burm.d: $(BUILD_OBJ_DIR)/Depend/.dir
	touch $@

SparcV9CodeEmitter.cpp: SparcV9CodeEmitter.inc

SparcV9CodeEmitter.inc: SparcV9.td SparcV9_F2.td SparcV9_F3.td SparcV9_F4.td SparcV9_Reg.td $(TBLGEN)
	@echo "TableGen-erating $@"
	cpp -P SparcV9.td | $(TBLGEN) -gen-emitter -o $@

clean::
	rm -f SparcV9CodeEmitter.inc
