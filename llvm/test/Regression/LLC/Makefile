#                    test/Regression/LLC/Makefile
#
# This directory contains regression tests for the LLVM sparc backend.  
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

# Only .ll tests here.  C tests are run via Makefile.singlesrc above.
# 
TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(filter-out %.c, $(TESTS:%.ll=%.ts)))

Output/%.ts: Output/%.bc $(LLC) Output/.dir
	@echo "======== Compiling $< =========="
	$(LLC) -f $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )
