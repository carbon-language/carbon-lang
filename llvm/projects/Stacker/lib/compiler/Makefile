##===- projects/sample/lib/sample/Makefile -----------------*- Makefile -*-===##

#
# Indicate where we are relative to the top of the source tree.
#
LEVEL=../../../..

#
# Give the name of a library.  This will build a dynamic version.
#
SHARED_LIBRARY=1
LIBRARYNAME=stkr_compiler

#
# Include Makefile.common so we know what to do.
#
include $(LEVEL)/Makefile.common

ifdef PARSE_DEBUG
INCLUDES += -DPARSE_DEBUG
endif
