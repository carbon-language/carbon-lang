#                              test/Feature/Makefile
#
# This makefile runs a moderate number of tests against LLVM source "feature"
# tests, which are designed to test individual components of the LLVM language.
# This runs through three sets of tests, designed to test to (dis)assembler,
# the optimizer, and the code generator for support for the more esoteric LLVM
# features.
#

LEVEL = ../..
include ../Makefile.tests

TESTS  := $(wildcard *.ll)
OTESTS := $(addprefix Output/, $(TESTS))  # Tests in output directory

test all :: testasmdis testopt testcbe testsparc
	@echo "All tests completed!"

testasmdis  : $(addsuffix .asmdis, $(OTESTS))
testopt     : $(addsuffix .opt   , $(OTESTS))

testcodegen : $(OTESTS:%.ll=%.mc)
testsparc   : $(OTESTS:%.ll=%.s)
testcbe     : $(OTESTS:%.ll=%.tc)

Output/%.asmdis: % $(LAS) $(LDIS) Output/.dir
	@./TestAsmDisasm.sh $< $(LLVMTOOLCURRENT) $(LIBDEBUG)

Output/%.opt: % $(LAS) $(LDIS) $(LOPT) Output/.dir
	@./TestOptimizer.sh $< $(LLVMTOOLCURRENT) $(LIBDEBUG)

Output/%.tc: Output/%.bc $(LDIS)
	@echo "======== Generating C code for $<"
	$(LDIS) -c < $< > $@    || \
                 ( rm -f $@; $(FAILURE) $@ )

Output/%.mc: Output/%.bc $(LLC)
	@echo "======== Generating machine instructions for $<"
	$(LLC) -f $(LLCFLAGS) $< > $@ || \
                 ( rm -f $@; $(FAILURE) $@ )

Output/%.s: Output/%.bc $(LLC)
	@echo "======== Generating assembly code for $<"
	$(LLC) -f $(LLCFLAGS) $< -o $@ || \
                 ( rm -f $@; $(FAILURE) $@ )

