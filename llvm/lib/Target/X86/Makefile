LEVEL = ../../..
LIBRARYNAME = x86
include $(LEVEL)/Makefile.common



# Make sure that tblgen is run, first thing.
$(SourceDepend): X86GenRegisterInfo.h.inc X86GenRegisterNames.inc X86GenRegisterInfo.inc

X86GenRegisterNames.inc:  $(wildcard *.td) $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-enums -o $@

X86GenRegisterInfo.h.inc: $(wildcard *.td) $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-desc-header -o $@

X86GenRegisterInfo.inc: $(wildcard *.td) $(TBLGEN)
	$(TBLGEN) X86.td -gen-register-desc -o $@

clean::
	$(VERB) rm -f *.inc

