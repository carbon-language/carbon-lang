#                    test/Regression/LLC/Makefile
#
# This directory contains regression tests for the LLVM sparc backend.  
#
LEVEL = ../../..
include $(LEVEL)/test/Programs/SingleSource/Makefile.singlesrc

TESTS := $(wildcard *.ll) $(wildcard *.c)

all:: $(addprefix Output/, $(filter-out %.c, $(TESTS:%.ll=%.ts)))

Output/%.ts: Output/%.bc $(LLC) Output/.dir
	@echo "======== Compiling $< =========="
	$(LLC) -f $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )
