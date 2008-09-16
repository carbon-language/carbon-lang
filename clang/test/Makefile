LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
TESTDIRS ?= $(shell echo $(PROJ_SRC_DIR)/*/)

# Only run rewriter tests on darwin.
ifeq ($(OS),Darwin)
TESTDIRS += 
endif

ifdef VERBOSE
PROGRESS = echo $<
REPORTFAIL = cat $@
DONE = true
else
PROGRESS = printf '.'
REPORTFAIL = (echo; echo '----' $< 'failed ----')
DONE = echo
endif

TESTS := $(addprefix Output/, $(addsuffix .testresults, $(shell find $(TESTDIRS) \( -name '*.c' -or -name '*.cpp' -or -name '*.m' -or -name '*.mm' \))))

Output/%.testresults: %
	@ $(PROGRESS)
	@ PATH=$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts:$$PATH VG=$(VG) $(PROJ_SRC_DIR)/TestRunner.sh $< > $@ || $(REPORTFAIL)

all::
	@ mkdir -p $(addprefix Output/, $(TESTDIRS))
	@ rm -f $(TESTS)
	@ echo '--- Running clang tests ---'
	@ $(MAKE) $(TESTS)
	@ $(DONE)
	@ cat $(TESTS) | grep -v 'XFAILED' | grep -v ' HAS NO RUN LINE! ****' | $(LLVM_SRC_ROOT)/test/Scripts/count 0 > /dev/null

report: $(TESTS)
	@ cat $^

clean::
	@ rm -rf Output/

.PHONY: all report clean
