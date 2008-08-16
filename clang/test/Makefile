LEVEL = ../../..
include $(LEVEL)/Makefile.common

TESTDIRS = CodeGen Lexer Preprocessor Parser Sema SemaCXX SemaObjC Analysis Serialization Rewriter

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

TESTS := $(addprefix Output/, $(addsuffix .testresults, $(shell find $(TESTDIRS) \( -name '*.c' -or -name '*.cpp' -or -name '*.m' \))))

Output/%.testresults: %
	@ $(PROGRESS)
	@ PATH=$$PATH:$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts VG=$(VG) ./TestRunner.sh $< > $@ || $(REPORTFAIL)

all::
	@ mkdir -p $(addprefix Output/, $(TESTDIRS))
	@ rm -f $(TESTS)
	@ echo '--- Running clang tests ---'
	@ $(MAKE) $(TESTS)
	@ $(DONE)

report: $(TESTS)
	@ cat $^

clean::
	@ rm -rf Output/

.PHONY: all report clean
