##===- lib/Target/Sparc/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
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
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/Ydefine/#define/' > $@

Sparc.burm : Sparc.burg.in1
	$(CXX) -E -I$(LLVM_SRC_ROOT)/include $(DEBUG_FLAG) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/^Xinclude/#include/' | $(SED) 's/^Xdefine/#define/' > $@

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

$(TARGET_NAME)CodeEmitter.cpp:: $(TARGET_NAME)CodeEmitter.inc

$(TARGET_NAME)CodeEmitter.inc:: $(TARGET_NAME).td $(TABLEGEN_FILES) $(TBLGEN)
	@echo "Tblgen'ing $<"
	$(TBLGEN) -I $(SourceDir) $< -gen-emitter -o $@

clean::
	$(RM) -f $(TARGET_NAME)CodeEmitter.inc Sparc.burg.in1 Sparc.burm Sparc.burm.cpp

