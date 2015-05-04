LEVEL = ../../make

C_SOURCES := main.c

ifeq "$(OS)" "Darwin"
    LDFLAGS = $(CFLAGS) -Xlinker -dead_strip
else
    CFLAGS += -fdata-sections -ffunction-sections
    LDFLAGS = $(CFLAGS) -Wl,--gc-sections
endif

MAKE_DSYM := NO

include $(LEVEL)/Makefile.rules
