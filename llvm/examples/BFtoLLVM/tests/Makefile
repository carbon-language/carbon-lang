##===- examples/BFtoLLVM/tests/Makefile --------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# Makefile for bf2llvm tests.

LEVEL = ../../..
BFTOLLVM = $(LLVMTOOLCURRENT)/BFtoLLVM

include $(LEVEL)/Makefile.common

all:: check

clean::
	rm -rf Output

.SUFFIXES: .ll .gccas.bc .llvm .cbe.c .cbe

Output/%.ll: %.b $(BFTOLLVM) Output/.dir
	$(BFTOLLVM) $< $@

Output/%.gccas.bc: Output/%.ll Output/.dir
	$(LGCCAS) $< -o $@

Output/%.llvm Output/%.llvm.bc: Output/%.gccas.bc Output/.dir
	$(LGCCLD) $< -lc -lcrtend -o Output/$*.llvm

Output/%.cbe.c: Output/%.llvm.bc Output/.dir
	$(LLC) -march=c -f -o=$@ $<

Output/%.cbe: Output/%.cbe.c Output/.dir
	$(CC) -O2 $< -o $@

check: Output/hello.cbe hello.expected-out
	@echo "Running test"
	Output/hello.cbe > Output/hello.out-cbe
	@echo "Checking result"
	diff Output/hello.out-cbe hello.expected-out
	@echo "Test passed"

