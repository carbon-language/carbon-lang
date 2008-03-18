LEVEL = ../../..
include $(LEVEL)/Makefile.common

TESTDIRS := CodeGen Lexer Preprocessor Parser Sema Analysis Serialization

# Only run rewriter tests on darwin.
ifeq ($(OS),Darwin)
TESTDIRS += Rewriter
endif

all::
	PATH=$$PATH:$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts \
          find $(TESTDIRS) \( -name '*.c' -or -name '*.cpp' -or -name '*.m' \) \
	      -print -exec ./TestRunner.sh {} \;

clean::
	rm -rf Output/

.PHONY: all clean
