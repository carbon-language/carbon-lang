LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
TESTDIRS ?= $(shell echo $(PROJ_SRC_DIR)/*/)

# Only run rewriter tests on darwin.
ifeq ($(OS),Darwin)
TESTDIRS += 
endif

ifdef VERBOSE
ifeq ($(VERBOSE),0)
PROGRESS = :
REPORTFAIL = echo 'FAIL: clang' $(TARGET_TRIPLE) $(subst $(LLVM_SRC_ROOT)/tools/clang/,,$<)
DONE = $(LLVMToolDir)/clang -v
else
PROGRESS = echo $<
REPORTFAIL = cat $@
DONE = true
endif
else
PROGRESS = printf '.'
REPORTFAIL = (echo; echo '----' $< 'failed ----')
DONE = echo
endif

TESTS := $(addprefix Output/, $(addsuffix .testresults, $(shell find $(TESTDIRS) \( -name '*.c' -or -name '*.cpp' -or -name '*.m' -or -name '*.mm' -or -name '*.S' \) | grep -v "Output/")))
Output/%.testresults: %
	@ $(PROGRESS)
	@ PATH=$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts:$$PATH VG=$(VG) $(PROJ_SRC_DIR)/TestRunner.sh $< > $@ || $(REPORTFAIL)

all::
	@ mkdir -p $(addprefix Output/, $(TESTDIRS))
	@ rm -f $(TESTS)
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ $(MAKE) $(TESTS)
	@ $(DONE)
	@ !(cat $(TESTS) | grep -q " FAILED! ")

report: $(TESTS)
	@ cat $^

clean::
	@ rm -rf Output/

.PHONY: all report clean
