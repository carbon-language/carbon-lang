###
# Configuration variables.

OS := $(shell uname)

# Assume make is always run from top-level of source directory. Note than an
# Apple style build overrides these variables later in the makefile.
ProjSrcRoot := $(shell pwd)
ProjObjRoot := $(ProjSrcRoot)

###
# Tool configuration variables.

CC := gcc
# FIXME: LLVM uses autoconf/mkinstalldirs ?
MKDIR := mkdir -p
DATE := date
AR := ar
# FIXME: Remove these pipes once ranlib errors are fixed.
AR.Flags := cru 2> /dev/null
RANLIB := ranlib
# FIXME: Remove these pipes once ranlib errors are fixed.
RANLIB.Flags := 2> /dev/null
LIPO := lipo
CP := cp

VERBOSE := 0
DEBUGMAKE :=

###
# Automatic and derived variables.

# Adjust settings for verbose mode
ifneq ($(VERBOSE),1)
  Verb := @
else
  Verb :=
endif

Echo := @echo
Archive := $(AR) $(AR.Flags)
Ranlib := $(RANLIB) $(RANLIB.Flags)
Lipo := $(LIPO)
ifndef Summary
	Summary = $(Echo)
endif
