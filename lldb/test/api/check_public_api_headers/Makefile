LEVEL = ../../make

CXX_SOURCES := main.cpp

MY_OS = $(shell uname -s)
ifeq "$(MY_OS)" "Darwin"
    LD_EXTRAS ?= -framework LLDB
else
    LD_EXTRAS ?= $(LLDB_LIB_DIR)/liblldb.so
endif

# If no c++ std was specified on CXXFLAGS, use c++11
ifeq (,$(findstring -std=c++,$(CXXFLAGS)))
  CFLAGS_EXTRAS := -std=c++11
endif

# Example dictionary to pass to the Python build method:
# 
# FRAMEWORK_INCLUDES=-F/Volumes/data/lldb/svn/trunk/build/Debug

include $(LEVEL)/Makefile.rules
