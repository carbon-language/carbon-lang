LEVEL = ../../..
include $(LEVEL)/Makefile.common

all::
	PATH=$$PATH:$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts find CodeGen Lexer Preprocessor Parser Sema \( -name '*.c' -or -name '*.cpp' \) -print -exec ./TestRunner.sh {} \;
