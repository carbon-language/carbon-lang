LEVEL = ../../..
LIBRARYNAME = x86
include $(LEVEL)/Makefile.common

# Make sure that tblgen is run, first thing.
$(SourceDepend): X86GenRegisterInfo.h.inc X86GenRegisterNames.inc \
                 X86GenRegisterInfo.inc X86GenInstrNames.inc \
                 X86GenInstrInfo.inc

X86GenRegisterNames.inc:  X86.td X86RegisterInfo.td $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-enums -o $@

X86GenRegisterInfo.h.inc: X86.td X86RegisterInfo.td $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-desc-header -o $@

X86GenRegisterInfo.inc: X86.td X86RegisterInfo.td $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-desc -o $@

X86GenInstrNames.inc: X86.td X86InstrInfo.td $(TBLGEN)
	$(TBLGEN) X86.td -gen-instr-enums -o $@

X86GenInstrInfo.inc: X86.td X86InstrInfo.td $(TBLGEN)
	$(TBLGEN) X86.td -gen-instr-desc -o $@

clean::
	$(VERB) rm -f *.inc

