#                    test/Regression/Assembler/Makefile
#
# This directory contains regression tests for the LLVM assembler program.  
# These LLVM source file tests are just required to assembler properly to pass.
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.out))

Output/%.out: %.ll $(LLC) Output/.dir
	@echo "======== Testing live variables for $<"
	$(LAS) < $< | $(LLC) -f -dlivevar i -o /dev/null 1> $@ 2>&1
	diff $@ . > /dev/null 2>& 1 || \
                 ( rm -f $@; $(FAILURE) $@ )

