###
# Configuration variables.

OS := $(shell uname)

# Assume make is always run from top-level of source directory. Note than an
# Apple style build overrides these variables later in the makefile.
ProjSrcRoot := $(shell pwd)
ProjObjRoot := $(ProjSrcRoot)

# The list of modules which are required to be built into every library. This
# should only be used for internal utilities which could be used in any other
# module. Any other cases the platform should be allowed to opt-in to.
AlwaysRequiredModules := int_util

###
# Tool configuration variables.

# FIXME: LLVM uses autoconf/mkinstalldirs ?
MKDIR := mkdir -p
DATE := date
LIPO := lipo
CP := cp
DSYMUTIL := dsymutil

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
ifndef Summary
  Summary = $(Echo)
endif

###
# Common compiler options
COMMON_INCLUDES=-I${ProjSrcRoot}/lib -I${ProjSrcRoot}/include
COMMON_CXXFLAGS=-std=c++11 -fno-exceptions -fPIC -funwind-tables $(COMMON_INCLUDES)
COMMON_CFLAGS=-fPIC $(COMMON_INCLUDES)
COMMON_ASMFLAGS=$(COMMON_INCLUDES)
