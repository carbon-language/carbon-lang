#                    test/Regression/Verifier/Makefile
#
# This directory contains regression tests for the LLVM verifier pass. Basically
# all of these LLVM source files are invalid programs that should be marked as
# such by the verifier

LEVEL = ../../..
include $(LEVEL)/test/Makefile.tests

TESTS := $(wildcard *.ll)

all:: $(addprefix Output/, $(TESTS:%.ll=%.ntbc))

Output/%.ntbc: %.ll $(LAS) Output/.dir
	@echo "======== Verifying $< ==========="
	if $(LAS) -f $< -o /dev/null; \
	then $(FAILURE) $@; exit 1; \
	else touch $@; exit 0;\
	fi
