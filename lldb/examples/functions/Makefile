LEVEL = ../../test/make

CXX_SOURCES := main.cpp

MY_OS = $(shell uname -s)
CXXFLAGS += -std=gnu++11 -stdlib=libc++
ifeq "$(MY_OS)" "Darwin"
    LD_EXTRAS ?= -framework LLDB -Wl,-rpath,/Applications/Xcode.app/Contents/SharedFrameworks
	FRAMEWORK_INCLUDES=-F/Applications/Xcode.app/Contents/SharedFrameworks
else
    LD_EXTRAS ?= $(LLDB_BUILD_DIR)/_lldb.so
endif

include $(LEVEL)/Makefile.rules
