
LEVEL = ../../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.llx)

all:: $(addprefix Output/, $(TESTS:%.llx=%.llx.out))

Output/%.llx.out: %.llx Output/.dir $(LOPT)
	-$(TESTRUNR) $<
