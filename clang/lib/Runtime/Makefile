##===- clang/lib/Runtime/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This file defines support for building the Clang runtime libraries (which are
# implemented by compiler-rt) and placing them in the proper locations in the
# Clang resources directory (i.e., where the driver expects them).
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
include $(LEVEL)/Makefile.common

CLANG_VERSION := $(shell cat $(PROJ_SRC_DIR)/../../VER)
ResourceDir := $(PROJ_OBJ_ROOT)/$(BuildMode)/lib/clang/$(CLANG_VERSION)
PROJ_resources := $(DESTDIR)$(PROJ_prefix)/lib/clang/$(CLANG_VERSION)

ResourceLibDir := $(ResourceDir)/lib
PROJ_resources_lib := $(PROJ_resources)/lib

# Expect compiler-rt to be in llvm/projects/compiler-rt
COMPILERRT_SRC_ROOT := $(LLVM_SRC_ROOT)/projects/compiler-rt

ifndef CLANG_NO_RUNTIME
ifeq ($(shell test -d $(COMPILERRT_SRC_ROOT) && echo OK),OK)

# Select the compiler-rt configuration to use, and install directory.
#
# FIXME: Eventually, we want some kind of configure support for this. We want to
# build/install runtime libraries for as many targets as clang was configured to
# support.
RuntimeDirs :=
ifeq ($(OS),Darwin)
RuntimeDirs += darwin
RuntimeLibrary.darwin.Configs = 10.4 armv6 cc_kext
endif

# Rule to build the compiler-rt libraries we need.
#
# We build all the libraries in a single shot to avoid recursive make as much as
# possible.
BuildRuntimeLibraries:
	$(Verb) $(MAKE) -C $(COMPILERRT_SRC_ROOT) \
	  ProjSrcRoot=$(COMPILERRT_SRC_ROOT) \
	  ProjObjRoot=$(PROJ_OBJ_DIR) \
	  $(RuntimeDirs:%=clang_%)
.PHONY: BuildRuntimeLibraries
CleanRuntimeLibraries:
	$(Verb) $(MAKE) -C $(COMPILERRT_SRC_ROOT) \
	  ProjSrcRoot=$(COMPILERRT_SRC_ROOT) \
	  ProjObjRoot=$(PROJ_OBJ_DIR) \
	  clean
.PHONY: CleanRuntimeLibraries

$(PROJ_resources_lib):
	$(Verb) $(MKDIR) $@

# Expand rules for copying/installing each individual library. We can't use
# implicit rules here because we need to match against multiple things.
define RuntimeLibraryTemplate
$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a: BuildRuntimeLibraries
	@true
.PRECIOUS: $(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a

# Rule to copy the libraries to their resource directory location.
$(ResourceLibDir)/$1/libclang_rt.%.a: \
		$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a \
		$(ResourceLibDir)/$1/.dir
	$(Echo) Copying runtime library $1/$$* to build dir
	$(Verb) cp $(PROJ_OBJ_DIR)/clang_$1/$$*/libcompiler_rt.a $$@
RuntimeLibrary.$1: \
		$(RuntimeLibrary.$1.Configs:%=$(ResourceLibDir)/$1/libclang_rt.%.a)
.PHONY: RuntimeLibrary.$1

$(PROJ_resources_lib)/$1: $(PROJ_resources_lib)
	$(Verb) $(MKDIR) $$@

$(PROJ_resources_lib)/$1/libclang_rt.%.a: \
		$(ResourceLibDir)/$1/libclang_rt.%.a | $(PROJ_resources_lib)/$1
	$(Echo) Installing compiler runtime library: $1/$$*
	$(Verb) $(DataInstall) $$< $(PROJ_resources_lib)/$1

# Rule to install runtime libraries.
RuntimeLibraryInstall.$1: \
		$(RuntimeLibrary.$1.Configs:%=$(PROJ_resources_lib)/$1/libclang_rt.%.a)
.PHONY: RuntimeLibraryInstall.$1
endef
$(foreach lib,$(RuntimeDirs), $(eval $(call RuntimeLibraryTemplate,$(lib))))

# Hook into the standard Makefile rules.
all-local:: $(RuntimeDirs:%=RuntimeLibrary.%)
install-local:: $(RuntimeDirs:%=RuntimeLibraryInstall.%)
clean-local:: CleanRuntimeLibraries

endif
endif
