#                    test/Regression/Assembler/Makefile
#
# This directory contains regression tests for the LLVM assembler program.  
# These LLVM source file tests are just required to assembler properly to pass.
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.tbc))

Output/%.tbc: %.ll $(LAS) Output/.dir
	@echo "======== Assembling $<"
	$(LAS) -f $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )
