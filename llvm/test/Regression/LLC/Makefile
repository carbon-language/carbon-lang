#                    test/Regression/LLC/Makefile
#
# This directory contains regression tests for the LLVM sparc backend.  
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

# Only .ll tests here.  C tests are run via Makefile.singlesrc above.
# 
LL_TESTS := $(wildcard *.ll)
LLX_TESTS := $(wildcard *.llx)

all:: $(addprefix Output/, $(LL_TESTS:%.ll=%.ts)) 
all:: $(addprefix Output/, $(LLX_TESTS:%.llx=%.llx.out))

Output/%.ts: Output/%.bc $(LLC) Output/.dir
	@echo "======== Compiling $< =========="
	$(LLC) -f $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )

Output/%.llx.out: %.llx Output/.dir $(LLC)
	-$(TESTRUNR) $<
