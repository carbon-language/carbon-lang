##===- lib/Target/PowerPC/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = powerpc
include $(LEVEL)/Makefile.common

# Make sure that tblgen is run, first thing.
$(SourceDepend): PowerPCGenRegisterInfo.h.inc PowerPCGenRegisterNames.inc \
                 PowerPCGenRegisterInfo.inc PowerPCGenInstrNames.inc \
                 PowerPCGenInstrInfo.inc PowerPCGenInstrSelector.inc

PowerPCGenRegisterNames.inc:: $(SourceDir)/PowerPC.td \
                           $(SourceDir)/PowerPCReg.td \
                           $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

PowerPCGenRegisterInfo.h.inc:: $(SourceDir)/PowerPC.td \
                           $(SourceDir)/PowerPCReg.td \
                           $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

PowerPCGenRegisterInfo.inc:: $(SourceDir)/PowerPC.td \
                         $(SourceDir)/PowerPCReg.td \
                         $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td register information implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

PowerPCGenInstrNames.inc:: $(SourceDir)/PowerPC.td \
                       $(SourceDir)/PowerPCInstrs.td \
                       $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

PowerPCGenInstrInfo.inc:: $(SourceDir)/PowerPC.td \
                      $(SourceDir)/PowerPCInstrs.td \
                      $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

PowerPCGenInstrSelector.inc:: $(SourceDir)/PowerPC.td \
                          $(SourceDir)/PowerPCInstrs.td \
                          $(SourceDir)/../Target.td $(TBLGEN)
	@echo "Building PowerPC.td instruction selector with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-selector -o $@

clean::
	$(VERB) rm -f *.inc
