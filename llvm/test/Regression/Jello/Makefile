#                    test/Regression/Jello/Makefile
#
# This directory contains regression tests for the LLVM jello program.  
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)
FTESTS := $(wildcard *.llx)   # Freeform tests

all:: $(addprefix Output/, $(TESTS:%.ll=%.out))

Output/%.out: Output/%.bc
	@echo "======== Running $< ==================="
	$(VERB) jello $< > $@ 2>&1 || \
                 ( cat $@; rm -f $@; $(FAILURE) $@ )


all:: $(addprefix Output/, $(FTESTS:%.llx=%.llx.out))

Output/%.llx.out: %.llx Output/.dir $(LAS)
	-$(TESTRUNR) $<

