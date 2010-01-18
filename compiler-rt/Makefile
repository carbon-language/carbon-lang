SubDirs := lib

# Set default rule before anything else.
all::

include make/config.mk
include make/util.mk
# If SRCROOT is defined, assume we are doing an Apple style build. We should be
# able to use RC_XBS for this but that is unused during "make installsrc".
ifdef SRCROOT
  include make/AppleBI.mk
endif

# Make sure we don't build with a missing ProjObjRoot.
ifeq ($(ProjObjRoot),)
$(error Refusing to build with empty ProjObjRoot variable)
endif

##############

###
# Rules

###
# Top level targets

# FIXME: Document the available subtargets.
help:
	@echo "usage: make [{VARIABLE=VALUE}*] target"
	@echo
	@echo "User variables:"
	@echo "  VERBOSE=1: Use to show all commands [default=0]"
	@echo
	@echo "Available targets:"
	@echo "  clean: clean all configurations"
	@echo "  test:  run unit tests"
	@echo "  all:   build all configurations"
	@echo

help-devel: help
	@echo "Development targets:"
	@echo "  info-functions: list available compiler-rt functions"
	@echo

help-hidden: help-devel
	@echo "Debugging variables:"
	@echo "  DEBUGMAKE=1: enable some Makefile logging [default=]"
	@echo "           =2: enable more Makefile logging"
	@echo
	@echo "Debugging targets:"
	@echo "  make-print-FOO: print information on the variable 'FOO'"
	@echo

info-functions:
	@echo "compiler-rt Available Functions"
	@echo
	@echo "All Functions: $(AvailableFunctions)"
	@$(foreach fn,$(AvailableFunctions),\
	  printf "  %-20s - available in (%s)\n" $(fn)\
	    "$(foreach key,$(AvailableIn.$(fn)),$($(key).Dir))";)

# Provide default clean target which is extended by other templates.
.PHONY: clean
clean::

# Test
.PHONY: test
test:
	cd test/Unit && ./test

###
# Directory handling magic.

# Create directories as needed, and timestamp their creation.
%/.dir:
	$(Summary) "  MKDIR:     $*"
	$(Verb) $(MKDIR) $* > /dev/null
	$(Verb) $(DATE) > $@

# Remove directories
%/.remove:
	$(Verb) $(RM) -r $*

###
# Include child makefile fragments

Dir := .
include make/subdir.mk
include make/lib_info.mk
include make/lib_util.mk

ifneq ($(DEBUGMAKE),)
  $(info MAKE: Done processing Makefile)
  $(info  )
endif
