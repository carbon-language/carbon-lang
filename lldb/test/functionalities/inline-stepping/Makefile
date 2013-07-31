LEVEL = ../../make

CXX_SOURCES := calling.cpp

ifneq (,$(findstring icc,$(CC)))
    CXXFLAGS += -debug inline-debug-info
endif

include $(LEVEL)/Makefile.rules
