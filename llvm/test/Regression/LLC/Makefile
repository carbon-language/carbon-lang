#                    test/Regression/LLC/Makefile
#
# This directory contains regression tests for the LLVM sparc backend.  
#
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.ts))

Output/%.ts: Output/%.bc $(LLC) Output/.dir
	@echo "======== Compiling $< =========="
	$(LLC) -f $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )
