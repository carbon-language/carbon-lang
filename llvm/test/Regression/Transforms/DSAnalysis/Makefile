#                    test/Regression/Transforms/DSAnalysis/Makefile
#
# This directory contains test cases for the llvm data structure analysis 
# library.  These are meant to test all of the nooks and cranies of the 
# algorithm, although we really have no way to verify that the answers produced
# are correct.
#
LEVEL = ../../../..
include $(LEVEL)/test/Makefile.tests

## Get absolute pathname for analyze
LANALYZE_ABS := $(shell cd `dirname $(LANALYZE)`; pwd)/analyze

TESTS := $(wildcard *.ll)
FTESTS := $(wildcard *.llx)   # Freeform tests

all:: $(addprefix Output/, $(TESTS:%.ll=%.out))

Output/%.out: %.ll $(LANALYZE) Output/.dir
	(cd Output/; $(LANALYZE_ABS) -datastructure ../$< || \
                 ( rm -f ../$@; ../$(FAILURE) ../$@ ))

all:: $(addprefix Output/, $(FTESTS:%.llx=%.llx.out))

clean::
	rm -f *.ps *.dot

Output/%.llx.out: %.llx Output/.dir $(LAS)
	-$(TESTRUNR) $<

