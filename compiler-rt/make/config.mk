###
# Configuration variables.

OS := $(shell uname)

# Assume make is always run from top-level of source directory. Note
# than an Apple style build overrides these variables later in the
# makefile.
ProjSrcRoot := $(shell pwd)
ProjObjRoot := $(ProjSrcRoot)

Configs := Debug Release Profile

# The full list of architectures we support.
Archs := i386 ppc x86_64

# If TargetArch is defined, only build for that architecture (and don't use
# -arch).
ifeq ($(OS), Darwin)
  TargetArch :=
  TargetArchs := $(Archs)
else
  TargetArch := i386
  TargetArchs := $(TargetArch)
endif

Common.CFLAGS := -Wall -Werror

# These names must match the configs, see GetArgs function.
Debug.CFLAGS := -g
Release.CFLAGS := -O3 -fomit-frame-pointer
Profile.CFLAGS := -pg -g

# Function: GetArchArgs arch
#
# Return the compiler flags for the given arch.
ifeq ($(OS), Darwin)
  GetArchArgs = -arch $(1)
else
  # Check that we are only trying to build the target arch.
  GetArchArgs = $(if $(subst $(TargetArch),,$(1)), \
	$(error "Invalid configuration, no -arch support: $(1)"), \
	)
endif

# Function: GetArgs config arch
#
# Return the compiler flags for the given config & arch.
GetArgs = $(if $($(1).CFLAGS), \
	        $(Common.CFLAGS) $($(1).CFLAGS) $(call GetArchArgs,$(2)), \
		$(error "Invalid configuration: $(1)"))

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

###
# Automatic and derived variables.

# Adjust settings for verbose mode
ifndef VERBOSE
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
