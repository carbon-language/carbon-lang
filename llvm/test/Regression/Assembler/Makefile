#                    test/Regression/Assembler/Makefile
#
# This directory contains regression tests for the LLVM assembler program.  
# These LLVM source file tests are just required to assembler properly to pass.
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)
FTESTS := $(wildcard *.llx)   # Freeform tests

all:: $(addprefix Output/, $(TESTS:%.ll=%.tbc))

Output/%.tbc: %.ll $(LAS) Output/.dir
	@echo "======== Assembling $<"
	$(LAS) -d -f $< -o $@ > /dev/null 2>&1 || \
                 ( rm -f $@; $(FAILURE) $@ )

all:: $(addprefix Output/, $(FTESTS:%.llx=%.ll.out))

Output/%.ll.out: %.llx Output/.dir $(LAS)
	-$(TESTRUNR) $<

