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

# Make sure that tblgen is run, first thing.
$(SourceDepend): SparcV8GenRegisterInfo.h.inc SparcV8GenRegisterNames.inc \
                 SparcV8GenRegisterInfo.inc SparcV8GenInstrNames.inc \
                 SparcV8GenInstrInfo.inc SparcV8GenInstrSelector.inc

SparcV8GenRegisterNames.inc:: $(SourceDir)/SparcV8.td \
                           $(SourceDir)/SparcV8Reg.td \
                           $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

SparcV8GenRegisterInfo.h.inc:: $(SourceDir)/SparcV8.td \
                           $(SourceDir)/SparcV8Reg.td \
                           $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

SparcV8GenRegisterInfo.inc:: $(SourceDir)/SparcV8.td \
                         $(SourceDir)/SparcV8Reg.td \
                         $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td register information implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

SparcV8GenInstrNames.inc:: $(SourceDir)/SparcV8.td \
                       $(SourceDir)/SparcV8Instrs.td \
                       $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

SparcV8GenInstrInfo.inc:: $(SourceDir)/SparcV8.td \
                      $(SourceDir)/SparcV8Instrs.td \
                      $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

SparcV8GenInstrSelector.inc:: $(SourceDir)/SparcV8.td \
                          $(SourceDir)/SparcV8Instrs.td \
                          $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building SparcV8.td instruction selector with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-selector -o $@

clean::
	$(VERB) rm -f *.inc
