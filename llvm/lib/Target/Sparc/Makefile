LEVEL = ../../..
LIBRARYNAME = sparc

ExtraSource = Sparc.burm.cpp 

include $(LEVEL)/Makefile.common

ifdef ENABLE_OPTIMIZED
  DEBUG_FLAG = 
else
  DEBUG_FLAG = -D_DEBUG
endif

Sparc.burg.in1 : Sparc.burg.in
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Ydefine/#define/' > $@

Sparc.burm : Sparc.burg.in1
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Xinclude/#include/g' | ${SED} 's/Xdefine/#define/g' > $@

Sparc.burm.cpp: Sparc.burm
	@echo "Burging $<"
	$(RunBurg) $< -o $@

$(BUILD_OBJ_DIR)/Debug/Sparc.burm.lo: Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_OBJ_DIR)/Release/Sparc.burm.lo: Sparc.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_OBJ_DIR)/Profile/Sparc.burm.lo: Sparc.burm.cpp
	$(CompileP) $< -o $@

$(BUILD_OBJ_DIR)/Depend/Sparc.burm.d: $(BUILD_OBJ_DIR)/Depend/.dir
	touch $@

TARGET_NAME := SparcV9

TABLEGEN_FILES := $(notdir $(wildcard $(SourceDir)/*.td))

$(SourceDir)/$(TARGET_NAME)CodeEmitter.cpp:: $(TARGET_NAME)CodeEmitter.inc

$(TARGET_NAME)CodeEmitter.inc:: $(TABLEGEN_FILES) $(TBLGEN)
	@echo "Tblgen'ing $(TARGET_NAME).td"
	$(TBLGEN) -I $(SourceDir) $(SourceDir)/$(TARGET_NAME).td -gen-emitter -o $@

clean::
	${RM} -f $(TARGET_NAME)CodeEmitter.inc Sparc.burg.in1 Sparc.burm Sparc.burm.cpp

