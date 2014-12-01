THIS_FILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/

LEVEL := $(realpath $(THIS_FILE_DIR)../../make)

CFLAGS_EXTRAS := -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS
ENABLE_THREADS := YES
CXX_SOURCES := $(wildcard *.cpp) \
	$(realpath $(LEVEL)/../../source/Utility/StringExtractor.cpp)
MAKE_DSYM := NO

OS := $(shell uname -s)
# $(info OS $(OS))
ifeq ($(OS),Linux)
    LD_EXTRAS := -lncurses -ldl
endif

include $(LEVEL)/Makefile.rules
