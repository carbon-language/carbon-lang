
LEVEL = ../../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.ibc))

Output/%.ibc: %.ll Output/.dir $(LOPT)
	($(LAS) < $< | $(LOPT) -inline -q -f -o $@) || \
	    ( rm -f $@; $(FAILURE) $@ )

