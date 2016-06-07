LEVEL = ../../../make
OBJC_SOURCES := main.m
LDFLAGS = $(CFLAGS) -lobjc -framework Foundation
ifneq (,$(findstring clang,$(CC)))
    CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules


cleanup:
	rm -f Makefile *.d

