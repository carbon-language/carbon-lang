##===- lib/Target/Skeleton/Makefile ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = skeleton
include $(LEVEL)/Makefile.common

TARGET = Skeleton
TDFILES = $(SourceDir)/$(TARGET).td $(wildcard $(SourceDir)/*.td) \
          $(SourceDir)/../Target.td


# Make sure that tblgen is run, first thing.
$(SourceDepend): $(TARGET)GenRegisterInfo.h.inc $(TARGET)GenRegisterNames.inc \
                 $(TARGET)GenRegisterInfo.inc $(TARGET)GenInstrNames.inc \
                 $(TARGET)GenInstrInfo.inc

$(TARGET)GenRegisterNames.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

$(TARGET)GenRegisterInfo.h.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

$(TARGET)GenRegisterInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td register information implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

$(TARGET)GenInstrNames.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

$(TARGET)GenInstrInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

clean::
	$(VERB) rm -f *.inc
