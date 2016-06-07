LEVEL = ../../make
CXX_SOURCES := main.cpp
CXXFLAGS += -std=c++11
ifneq (,$(findstring clang,$(CC)))
    CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules


cleanup:
	rm -f Makefile *.d

