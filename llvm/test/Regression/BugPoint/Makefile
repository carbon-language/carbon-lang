#                    test/Regression/Assembler/Makefile
#
# This directory contains regression tests for the LLVM assembler program.  
# These LLVM source file tests are just required to assembler properly to pass.
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.ll.out))

Output/%.ll.out: %.ll Output/.dir $(LAS) $(LBUGPOINT)
	-$(TESTRUNR) $<
	-@# Remove temporary files generated...
	-@rm -f bugpoint-*

clean::
	rm -f *.bc
