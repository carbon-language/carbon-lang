LEVEL = ../../..
LIBRARYNAME = x86
include $(LEVEL)/Makefile.common

# Make sure that tblgen is run, first thing.
$(SourceDepend): X86GenRegisterInfo.h.inc X86GenRegisterNames.inc \
                 X86GenRegisterInfo.inc X86GenInstrNames.inc \
                 X86GenInstrInfo.inc X86GenInstrSelector.inc

X86GenRegisterNames.inc::  X86.td X86RegisterInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-enums -o $@

X86GenRegisterInfo.h.inc:: X86.td X86RegisterInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc-header -o $@

X86GenRegisterInfo.inc:: X86.td X86RegisterInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-register-desc -o $@

X86GenInstrNames.inc:: X86.td X86InstrInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-enums -o $@

X86GenInstrInfo.inc:: X86.td X86InstrInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-desc -o $@

X86GenInstrSelector.inc:: X86.td X86InstrInfo.td ../Target.td $(TBLGEN)
	$(TBLGEN) -I $(BUILD_SRC_DIR) $< -gen-instr-selector -o $@

clean::
	$(VERB) rm -f *.inc

