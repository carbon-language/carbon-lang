#===- ./Makefile -------------------------------------------*- Makefile -*--===#
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#
LEVEL = .
DIRS = lib/System lib/Support utils lib

include $(LEVEL)/Makefile.common

ifeq ($(MAKECMDGOALS),tools-only)
  DIRS += tools
else
  ifneq ($(MAKECMDGOALS),libs-only)
    DIRS += tools
    ifneq ($(LLVMGCC_MAJVERS),4)
      DIRS += runtime
    else
      $(warning Skipping runtime libraries, llvm-gcc 4 detected.)
    endif
    DIRS += docs
    OPTIONAL_DIRS = examples projects
  endif
endif
EXTRA_DIST := test llvm.spec include

# Specify options to pass to configure script when we're
# running the dist-check target
DIST_CHECK_CONFIG_OPTIONS = --with-llvmgccdir=$(LLVMGCCDIR)

.PHONY: debug-opt-prof
debug-opt-prof:
	$(Echo) Building Debug Version
	$(Verb) $(MAKE)
	$(Echo)
	$(Echo) Building Optimized Version
	$(Echo)
	$(Verb) $(MAKE) ENABLE_OPTIMIZED=1
	$(Echo)
	$(Echo) Building Profiling Version
	$(Echo)
	$(Verb) $(MAKE) ENABLE_PROFILING=1

dist-hook::
	$(Echo) Eliminating files constructed by configure
	$(Verb) $(RM) -f \
	  $(TopDistDir)/include/llvm/ADT/hash_map  \
	  $(TopDistDir)/include/llvm/ADT/hash_set  \
	  $(TopDistDir)/include/llvm/ADT/iterator  \
	  $(TopDistDir)/include/llvm/Config/config.h  \
	  $(TopDistDir)/include/llvm/Support/DataTypes.h  \
	  $(TopDistDir)/include/llvm/Support/ThreadSupport.h

tools-only: all
libs-only: all

#------------------------------------------------------------------------
# Make sure the generated headers are up-to-date. This must be kept in
# sync with the AC_CONFIG_HEADER invocations in autoconf/configure.ac
#------------------------------------------------------------------------
FilesToConfig := \
  include/llvm/Config/config.h \
  include/llvm/Support/DataTypes.h \
  include/llvm/ADT/hash_map \
  include/llvm/ADT/hash_set \
  include/llvm/ADT/iterator
FilesToConfigPATH  := $(addprefix $(LLVM_OBJ_ROOT)/,$(FilesToConfig))

all-local:: $(FilesToConfigPATH)
$(FilesToConfigPATH) : $(LLVM_OBJ_ROOT)/% : $(LLVM_SRC_ROOT)/%.in 
	$(Echo) Regenerating $*
	$(Verb) cd $(LLVM_OBJ_ROOT) && $(ConfigStatusScript) $*
.PRECIOUS: $(FilesToConfigPATH)

# NOTE: This needs to remain as the last target definition in this file so
# that it gets executed last.
all:: 
	$(Echo) '*****' Completed $(BuildMode)$(AssertMode) Build
ifeq ($(BuildMode),Debug)
	$(Echo) '*****' Note: Debug build can be 10 times slower than an
	$(Echo) '*****' optimized build. Use 'make ENABLE_OPTIMIZED=1' to
	$(Echo) '*****' make an optimized build.
endif

