##===- lib/Target/X86/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = x86
include $(LEVEL)/Makefile.common

TARGET = X86

# Make sure that tblgen is run, first thing.
$(SourceDepend): X86GenRegisterInfo.h.inc X86GenRegisterNames.inc \
                 X86GenRegisterInfo.inc X86GenInstrNames.inc \
                 X86GenInstrInfo.inc X86GenAsmWriter.inc

TDFILES = $(SourceDir)/$(TARGET).td $(wildcard $(SourceDir)/*.td) \
          $(SourceDir)/../Target.td

$(TARGET)GenRegisterNames.inc::  $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

$(TARGET)GenRegisterInfo.h.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

$(TARGET)GenRegisterInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register info implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

$(TARGET)GenInstrNames.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

$(TARGET)GenInstrInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

$(TARGET)GenAsmWriter.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td assembly writer with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-asm-writer -o $@

#$(TARGET)GenInstrSelector.inc:: $(TDFILES) $(TBLGEN)
#	@echo "Building $(TARGET).td instruction selector with tblgen"
#	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-selector -o $@

clean::
	$(VERB) rm -f *.inc
