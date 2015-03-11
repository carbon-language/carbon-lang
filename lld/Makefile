##===- Makefile --------------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# If LLD_LEVEL is not set, then we are the top-level Makefile. Otherwise, we
# are being included from a subdirectory makefile.

ifndef LLD_LEVEL

IS_TOP_LEVEL := 1
LLD_LEVEL := .
DIRS := include lib tools unittests

PARALLEL_DIRS :=

endif

ifeq ($(MAKECMDGOALS),libs-only)
  DIRS := $(filter-out tools docs, $(DIRS))
  OPTIONAL_DIRS :=
endif
ifeq ($(BUILD_LLD_ONLY),YES)
  DIRS := $(filter-out docs unittests, $(DIRS))
  OPTIONAL_DIRS :=
endif

###
# Common Makefile code, shared by all lld Makefiles.

# Set LLVM source root level.
LEVEL := $(LLD_LEVEL)/../..

# Include LLVM common makefile.
include $(LEVEL)/Makefile.common

ifneq ($(ENABLE_DOCS),1)
  DIRS := $(filter-out docs, $(DIRS))
endif

CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLD_LEVEL)/include
CPP.Flags += -I$(PROJ_OBJ_DIR)/$(LLD_LEVEL)/include

###
# lld Top Level specific stuff.

ifeq ($(IS_TOP_LEVEL),1)

ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
$(RecursiveTargets)::
	$(Verb) for dir in test unittests; do \
	  if [ -f $(PROJ_SRC_DIR)/$${dir}/Makefile ] && [ ! -f $${dir}/Makefile ]; then \
	    $(MKDIR) $${dir}; \
	    $(CP) $(PROJ_SRC_DIR)/$${dir}/Makefile $${dir}/Makefile; \
	  fi \
	done
endif

test::
	@ $(MAKE) -C test

report::
	@ $(MAKE) -C test report

clean::
	@ $(MAKE) -C test clean

libs-only: all

tags::
	$(Verb) etags `find . -type f -name '*.h' -or -name '*.cpp' | \
	  grep -v /lib/Headers | grep -v /test/`

cscope.files:
	find tools lib include -name '*.cpp' \
	                    -or -name '*.def' \
	                    -or -name '*.td' \
	                    -or -name '*.h' > cscope.files

.PHONY: test report clean cscope.files

endif
