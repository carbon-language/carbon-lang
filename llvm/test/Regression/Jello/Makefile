#                    test/Regression/Jello/Makefile
#
# This directory contains regression tests for the LLVM x86 JIT  
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)
FTESTS := $(wildcard *.llx)   # Freeform tests

all:: $(addprefix Output/, $(TESTS:%.ll=%.out))

LLI_FLAGS = -force-interpreter=false

ifdef DISABLE_FP_ELIM
LLI_FLAGS += -disable-fp-elim
endif

Output/%.out: Output/%.bc $(LLI)
	@echo "======== Running $< ==================="
	$(VERB) $(LLI) $(LLI_FLAGS) -stats $< > $@ 2>&1 || \
                 ( cat $@; rm -f $@; $(FAILURE) $@ )


all:: $(addprefix Output/, $(FTESTS:%.llx=%.llx.out))

Output/%.llx.out: %.llx Output/.dir $(LJELLO)
	-$(TESTRUNR) $<

