
LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.td)

all:: $(addprefix Output/, $(TESTS:%=%.out))

Output/%.out: % Output/.dir $(TBLGEN)
	-$(TESTRUNR) $<
