##===- lib/Target/SparcV8/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = sparcv8
include $(LEVEL)/Makefile.common

TDFILES := $(wildcard $(SourceDir)/*.td) $(SourceDir)/../Target.td
TDFILE  := $(SourceDir)/SparcV8.td

# Make sure that tblgen is run, first thing.
$(SourceDepend): SparcV8GenRegisterInfo.h.inc SparcV8GenRegisterNames.inc \
                 SparcV8GenRegisterInfo.inc SparcV8GenInstrNames.inc \
                 SparcV8GenInstrInfo.inc SparcV8GenInstrSelector.inc

SparcV8GenRegisterNames.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-register-enums -o $@

SparcV8GenRegisterInfo.h.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-register-desc-header -o $@

SparcV8GenRegisterInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td register information implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-register-desc -o $@

SparcV8GenInstrNames.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-instr-enums -o $@

SparcV8GenInstrInfo.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-instr-desc -o $@

SparcV8GenInstrSelector.inc:: $(TDFILES) $(TBLGEN)
	@echo "Building SparcV8.td instruction selector with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $(TDFILE) -gen-instr-selector -o $@

clean::
	$(VERB) rm -f *.inc
