#            test/Regression/Transforms/DecomposeArrayRefs/Makefile
#
# This directory contains regression tests for the DecomposeArrayRefs pass.
# Since the main tests are single C source files, we'll use the same
# Makefile as test/Programs/SingleSource.
# The *.ll tests are mainly useful to run manually.
#
LEVEL = ../../../..
include $(LEVEL)/test/Programs/SingleSource/Makefile.singlesrc

LLTESTS := $(wildcard *.ll)
CTESTS  := $(wildcard *.c) 
TESTS   := $(LLTESTS) $(CTESTS)

all:: $(addprefix Output/, $(LLTESTS:%.ll=%.ll.out))

Output/%.ll.out: %.ll Output/.dir $(LOPT)
	-$(TESTRUNR) $<

