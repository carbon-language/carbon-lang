##===- projects/sample/lib/sample/Makefile -----------------*- Makefile -*-===##

#
# Indicate where we are relative to the top of the source tree.
#
LEVEL=../../../..

#
# Give the name of a library.  This will build a dynamic version.
#
TOOLNAME=stkrc
USEDLIBS=stkr_compiler asmparser bcwriter vmcore support.a

ifdef PARSE_DEBUG
CPPFLAGS = -DPARSE_DEBUG=1
endif

ifdef FLEX_DEBUG
CPPFLAGS += -DFLEX_DEBUG=1
endif

#
# Include Makefile.common so we know what to do.
#
include $(LEVEL)/Makefile.common

