##===- lib/Target/SparcV9/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = sparcv9
DIRS = InstrSelection RegAlloc LiveVar

ExtraSource = SparcV9.burm.cpp 

include $(LEVEL)/Makefile.common

ifdef ENABLE_OPTIMIZED
  DEBUG_FLAG = 
else
  DEBUG_FLAG = -D_DEBUG
endif

SparcV9.burg.in1 : $(SourceDir)/SparcV9.burg.in
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/Ydefine/#define/' > $@

SparcV9.burm : SparcV9.burg.in1
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/^Xinclude/#include/' | $(SED) 's/^Xdefine/#define/' > $@

SparcV9.burm.cpp: SparcV9.burm
	@echo "Burging `basename $<`"
	$(RunBurg) $< -o $@

$(BUILD_OBJ_DIR)/Debug/SparcV9.burm.lo: SparcV9.burm.cpp
	$(CompileG) $< -o $@

$(BUILD_OBJ_DIR)/Release/SparcV9.burm.lo: SparcV9.burm.cpp
	$(CompileO) $< -o $@

$(BUILD_OBJ_DIR)/Profile/SparcV9.burm.lo: SparcV9.burm.cpp
	$(CompileP) $< -o $@

$(BUILD_OBJ_DIR)/Depend/SparcV9.burm.d: $(BUILD_OBJ_DIR)/Depend/.dir
	touch $@

TARGET_NAME := SparcV9

TABLEGEN_FILES := $(notdir $(wildcard $(SourceDir)/*.td))

# Make sure that tblgen is run, first thing.
$(SourceDepend): $(TARGET_NAME)CodeEmitter.inc

$(TARGET_NAME)CodeEmitter.cpp:: $(TARGET_NAME)CodeEmitter.inc

$(TARGET_NAME)CodeEmitter.inc:: $(SourceDir)/$(TARGET_NAME).td $(TABLEGEN_FILES) $(TBLGEN)
	@echo "Tblgen'ing `basename $<`"
	$(TBLGEN) -I $(SourceDir) $< -gen-emitter -o $@

clean::
	$(RM) -f $(TARGET_NAME)CodeEmitter.inc SparcV9.burg.in1 SparcV9.burm SparcV9.burm.cpp

