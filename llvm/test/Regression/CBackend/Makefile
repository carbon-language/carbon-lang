#                    test/Regression/CBackend/Makefile
#
# This directory contains regression tests for the C backend for LLVM.  
# These testcases are assembled with the LLVM assembler, then disassembled into
# C code.  The C code should be compilable with the standard C compiler.
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

.PRECIOUS: Output/%.c

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.to))


Output/%.to: Output/%.c
	$(CC) -c $< -o $@ || \
		(rm -f $@; $(FAILURE) $@ )

Output/%.c: %.ll Output/.dir $(LAS) $(LDIS)
	$(LAS) < $< | $(LDIS) -c > $@ || \
		(rm -f $@; $(FAILURE) $@ )
