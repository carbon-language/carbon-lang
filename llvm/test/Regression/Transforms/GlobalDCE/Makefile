
LEVEL = ../../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.ll.out))

Output/%.ll.out: %.ll Output/.dir $(LOPT)
	-$(TESTRUNR) $<
