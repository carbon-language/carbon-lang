LEVEL = ../../..
LIBRARYNAME = sparc

ExtraSource = Debug/Sparc.burm.cpp 

include $(LEVEL)/Makefile.common

ifdef ENABLE_OPTIMIZED
  DEBUG_FLAG = 
else
  DEBUG_FLAG = -D_DEBUG
endif

Debug/Sparc.burg.in1 : $(SourceDir)/Sparc.burg.in Debug/.dir
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Ydefine/#define/' > $@

Debug/Sparc.burm : Debug/Sparc.burg.in1
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | ${SED} '/^# /d' | ${SED} 's/Xinclude/#include/g' | ${SED} 's/Xdefine/#define/g' > $@

Debug/Sparc.burm.cpp: Debug/Sparc.burm Debug/.dir
	$(RunBurg) $< -o $@

$(BUILD_OBJ_DIR)/Debug/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_OBJ_DIR)/Release/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_OBJ_DIR)/Profile/Sparc.burm.lo: Debug/Sparc.burm.cpp
	$(CompileP) $< -o $@

$(BUILD_OBJ_DIR)/Depend/Sparc.burm.d: $(BUILD_OBJ_DIR)/Depend/.dir
	touch $@

TARGET_NAME := SparcV9

TABLEGEN_FILES := $(wildcard $(SourceDir)/*.td)

$(SourceDir)/$(TARGET_NAME)CodeEmitter.cpp:: $(TARGET_NAME)CodeEmitter.inc

$(TARGET_NAME)CodeEmitter.inc:: $(TABLEGEN_FILES) $(TBLGEN)
	$(TBLGEN) -I $(SourceDir) $(SourceDir)/$(TARGET_NAME).td -gen-emitter -o $@

clean::
	${RM} -f $(TARGET_NAME)CodeEmitter.inc
