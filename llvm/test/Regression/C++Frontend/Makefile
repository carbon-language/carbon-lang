#             test/Regression/C++Frontend/CompileFail/Makefile
#
# This directory contains regression tests for the GCC C++ frontend for LLVM.  
# These tests are C++ source files that are input to GCC and compiled to .ll
# files. After that, the files are assembled by the LLVM 'as' program to make
# sure GCC emitted something valid.  This is intended to catch crashers and bad
# output by GCC.  Miscompilations by GCC should be caught by the 
# test/Programs/SingleSource/UnitTests test suite
#
LEVEL = ../../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.cpp)
TR_TESTS := $(wildcard *.cpp.tr)

all:: $(addprefix Output/, $(TESTS:%.cpp=%.tbc))
all:: $(addprefix Output/, $(TR_TESTS:%=%.out))

Output/%.cpp.tr.out: %.cpp.tr Output/.dir $(LCC1XX)
	-$(TESTRUNR) $<
