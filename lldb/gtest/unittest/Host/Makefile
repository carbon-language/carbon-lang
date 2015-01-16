THIS_FILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/

LEVEL := $(realpath $(THIS_FILE_DIR)../../make)

CFLAGS_EXTRAS := -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS
ENABLE_THREADS := YES
# the fact that we need all of these source files to compile Socket.cpp
# is a good indication that we need some refactoring
CXX_SOURCES := $(wildcard *.cpp) \
    $(realpath $(LEVEL)/../../source/Core/Error.cpp) \
    $(realpath $(LEVEL)/../../source/Core/RegularExpression.cpp) \
    $(realpath $(LEVEL)/../../source/Core/Stream.cpp) \
    $(realpath $(LEVEL)/../../source/Core/StreamString.cpp) \
    $(realpath $(LEVEL)/../../source/Host/common/Socket.cpp) \
    $(realpath $(LEVEL)/../../source/Host/common/SocketAddress.cpp) \
    $(realpath $(LEVEL)/../../source/Host/common/StringConvert.cpp) \
    $(realpath $(LEVEL)/../../source/Host/common/TimeValue.cpp)

OS := $(shell uname -s)
ifeq ($(OS),Windows)
CXX_SOURCES := $(CXX_SOURCES) \
    $(LEVEL)/../../source/Host/windows/Condition.cpp \
    $(LEVEL)/../../source/Host/windows/Mutex.cpp
else
CXX_SOURCES := $(CXX_SOURCES) \
    $(LEVEL)/../../source/Host/common/Condition.cpp \
    $(LEVEL)/../../source/Host/common/Mutex.cpp
endif

MAKE_DSYM := NO

include $(LEVEL)/Makefile.rules
