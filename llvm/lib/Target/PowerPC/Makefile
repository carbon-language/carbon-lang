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

TARGET = PowerPC

# Make sure that tblgen is run, first thing.
$(SourceDepend): PowerPCGenInstrNames.inc PowerPCGenRegisterNames.inc \
                 PowerPCGenCodeEmitter.inc PowerPCGenAsmWriter.inc \
 PPC32GenRegisterInfo.h.inc PPC32GenRegisterInfo.inc PPC32GenInstrInfo.inc \
 PPC64GenRegisterInfo.h.inc PPC64GenRegisterInfo.inc PPC64GenInstrInfo.inc

TDFILES = $(wildcard $(SourceDir)/*.td) $(SourceDir)/../Target.td

%GenRegisterNames.inc:: PPC32.td $(TDFILES) $(TBLGEN)
	@echo "Building PowerPC register names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

%GenRegisterInfo.h.inc:: %.td $(TDFILES) $(TBLGEN)
	@echo "Building `basename $<` register information header with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

%GenRegisterInfo.inc:: %.td $(TDFILES) $(TBLGEN)
	@echo "Building `basename $<` register information implementation with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

$(TARGET)GenInstrNames.inc:: PPC32.td $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET) instruction names with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

%GenInstrInfo.inc:: %.td $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET) instruction information with tblgen"
	$(VERB) $(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

$(TARGET)GenCodeEmitter.inc:: PPC32.td $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET) code emitter"
	$(VERB) $(TBLGEN) -I $(SourceDir) $< -gen-emitter -o $@

$(TARGET)GenAsmWriter.inc:: PowerPC.td $(TDFILES) $(TBLGEN)
	@echo "Building $(TARGET).td assembly writer with tblgen"
	$(VERB) $(TBLGEN) -I $(SourceDir) $< -gen-asm-writer -o $@

clean::
	$(VERB) rm -f *.inc
