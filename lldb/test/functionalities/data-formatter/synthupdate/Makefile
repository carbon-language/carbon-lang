LEVEL = ../../../make

OBJC_SOURCES := main.m

CFLAGS_EXTRAS += -w

include $(LEVEL)/Makefile.rules

LDFLAGS += -framework Foundation

clean::
	rm -rf $(wildcard *.o *.d *.dSYM *.log)
