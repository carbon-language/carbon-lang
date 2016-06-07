LEVEL = ../../../make
C_SOURCES := main.c
ifneq (,$(findstring clang,$(CC)))
    CFLAGS_EXTRAS += -fno-limit-debug-info
endif

include $(LEVEL)/Makefile.rules


cleanup:
	rm -f Makefile *.d

