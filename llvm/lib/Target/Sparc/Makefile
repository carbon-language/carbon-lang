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

$(BUILD_OBJ_DIR)/Debug/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_OBJ_DIR)/Release/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_OBJ_DIR)/Profile/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileP) $< -o $@

#$(BUILD_OBJ_DIR)/Debug/Sparc.burm.o: Debug/Sparc.burm.cpp
#	$(CompileG) $< -o $@

#$(BUILD_OBJ_DIR)/Release/Sparc.burm.o: Debug/Sparc.burm.cpp
#	$(CompileO) $< -o $@

#$(BUILD_OBJ_DIR)/Profile/Sparc.burm.o: Debug/Sparc.burm.cpp
#	$(CompileP) $< -o $@

Debug/Sparc.burg.in1 : Sparc.burg.in Debug/.dir
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Ydefine/#define/' > $@

Debug/Sparc.burm : Debug/Sparc.burg.in1
	$(CXX) -E -I$(LEVEL)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Xinclude/#include/g' | ${SED} 's/Xdefine/#define/g' > $@

$(BUILD_OBJ_DIR)/Depend/Sparc.burm.d: $(BUILD_OBJ_DIR)/Depend/.dir
	touch $@

SparcV9CodeEmitter.cpp: SparcV9CodeEmitter.inc


TARGET_NAME := SparcV9

TABLEGEN_FILES := $(wildcard *.td)

$(TARGET_NAME)CodeEmitter.inc.tmp: $(TABLEGEN_FILES) $(TBLGEN)
	@${ECHO} "TableGen-erating " $(TARGET_NAME)
	$(CC) -E -xc -P $(TARGET_NAME).td | $(TBLGEN) -gen-emitter -o $@

$(TARGET_NAME)CodeEmitter.inc: $(TARGET_NAME)CodeEmitter.inc.tmp
	$(VERB) # Only copy over the real .inc file if it has changed!
	$(VERB) diff $@ $< > /dev/null || cp $< $@

clean::
	${RM} -f $(TARGET_NAME)CodeEmitter.inc $(TARGET_NAME)CodeEmitter.inc.tmp
