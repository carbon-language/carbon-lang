#===- ./Makefile -------------------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

LEVEL := .

# Top-Level LLVM Build Stages:
#   1. Build lib/Support and lib/TableGen, which are used by utils (tblgen).
#   2. Build utils, which is used by IR.
#   3. Build IR, which builds the Intrinsics.inc file used by libs.
#   4. Build libs, which are needed by llvm-config.
#   5. Build llvm-config, which determines inter-lib dependencies for tools.
#   6. Build tools and docs.
#
# When cross-compiling, there are some things (tablegen) that need to
# be build for the build system first.

# If "RC_ProjectName" exists in the environment, and its value is
# "llvmCore", then this is an "Apple-style" build; search for
# "Apple-style" in the comments for more info.  Anything else is a
# normal build.
ifneq ($(findstring llvmCore, $(RC_ProjectName)),llvmCore)  # Normal build (not "Apple-style").

ifeq ($(BUILD_DIRS_ONLY),1)
  DIRS := lib/Support lib/TableGen utils tools/llvm-config
  OPTIONAL_DIRS := tools/clang/utils/TableGen
else
  DIRS := lib/Support lib/TableGen utils lib/IR lib tools/llvm-shlib \
          tools/llvm-config tools docs unittests
  OPTIONAL_DIRS := projects bindings
endif

ifeq ($(BUILD_EXAMPLES),1)
  OPTIONAL_DIRS += examples
endif

EXTRA_DIST := test unittests llvm.spec include win32 Xcode

include $(LEVEL)/Makefile.config

ifneq ($(ENABLE_SHARED),1)
  DIRS := $(filter-out tools/llvm-shlib, $(DIRS))
endif

ifneq ($(ENABLE_DOCS),1)
  DIRS := $(filter-out docs, $(DIRS))
endif

ifeq ($(MAKECMDGOALS),libs-only)
  DIRS := $(filter-out tools docs, $(DIRS))
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),install-libs)
  DIRS := $(filter-out tools docs, $(DIRS))
  OPTIONAL_DIRS := $(filter bindings, $(OPTIONAL_DIRS))
endif

ifeq ($(MAKECMDGOALS),tools-only)
  DIRS := $(filter-out docs, $(DIRS))
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),install-clang)
  DIRS := tools/clang/tools/driver tools/clang/lib/Headers \
          tools/clang/tools/libclang \
          tools/clang/tools/c-index-test \
          tools/clang/include/clang-c \
          tools/clang/runtime tools/clang/docs \
          tools/lto
  OPTIONAL_DIRS :=
  NO_INSTALL = 1
endif

ifeq ($(MAKECMDGOALS),clang-only)
  DIRS := $(filter-out tools docs unittests, $(DIRS)) \
          tools/clang tools/lto
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),unittests)
  DIRS := $(filter-out tools docs, $(DIRS)) utils unittests
  OPTIONAL_DIRS :=
endif

# Use NO_INSTALL define of the Makefile of each directory for deciding
# if the directory is installed or not
ifeq ($(MAKECMDGOALS),install)
  OPTIONAL_DIRS := $(filter bindings, $(OPTIONAL_DIRS))
endif

# Don't build unittests when ONLY_TOOLS is set.
ifneq ($(ONLY_TOOLS),)
  DIRS := $(filter-out unittests, $(DIRS))
endif

# If we're cross-compiling, build the build-hosted tools first
ifeq ($(LLVM_CROSS_COMPILING),1)
all:: cross-compile-build-tools

clean::
	$(Verb) rm -rf BuildTools

cross-compile-build-tools:
	$(Verb) if [ ! -f BuildTools/Makefile ]; then \
          $(MKDIR) BuildTools; \
	  cd BuildTools ; \
	  unset CFLAGS ; \
	  unset CXXFLAGS ; \
	  unset SDKROOT ; \
	  unset UNIVERSAL_SDK_PATH ; \
	  configure_opts= ; \
	  if test "$(ENABLE_LIBCPP)" -ne 0 ; then \
	    configure_opts="$$configure_opts --enable-libcpp"; \
	  fi; \
	  $(PROJ_SRC_DIR)/configure --build=$(BUILD_TRIPLE) \
		--host=$(BUILD_TRIPLE) --target=$(BUILD_TRIPLE) \
	        --disable-polly $$configure_opts; \
	  cd .. ; \
	fi; \
	($(MAKE) -C BuildTools \
	  BUILD_DIRS_ONLY=1 \
	  UNIVERSAL= \
	  UNIVERSAL_SDK_PATH= \
	  SDKROOT= \
	  TARGET_NATIVE_ARCH="$(TARGET_NATIVE_ARCH)" \
	  TARGETS_TO_BUILD="$(TARGETS_TO_BUILD)" \
	  TARGET_LIBS="$(LIBS)" \
	  ENABLE_OPTIMIZED=$(ENABLE_OPTIMIZED) \
	  ENABLE_PROFILING=$(ENABLE_PROFILING) \
	  ENABLE_COVERAGE=$(ENABLE_COVERAGE) \
	  DISABLE_ASSERTIONS=$(DISABLE_ASSERTIONS) \
	  ENABLE_EXPENSIVE_CHECKS=$(ENABLE_EXPENSIVE_CHECKS) \
	  ENABLE_LIBCPP=$(ENABLE_LIBCPP) \
	  CFLAGS= \
	  CXXFLAGS= \
	) || exit 1;
endif

# Include the main makefile machinery.
include $(LLVM_SRC_ROOT)/Makefile.rules

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
	  $(TopDistDir)/include/llvm/Config/config.h  \
	  $(TopDistDir)/include/llvm/Support/DataTypes.h

clang-only: all
tools-only: all
libs-only: all
install-clang: install
install-libs: install

# If SHOW_DIAGNOSTICS is enabled, clear the diagnostics file first.
ifeq ($(SHOW_DIAGNOSTICS),1)
clean-diagnostics:
	$(Verb) rm -f $(LLVM_OBJ_ROOT)/$(BuildMode)/diags
.PHONY: clean-diagnostics

all-local:: clean-diagnostics
endif

#------------------------------------------------------------------------
# Make sure the generated files are up-to-date. This must be kept in
# sync with the AC_CONFIG_HEADER and AC_CONFIG_FILE invocations in
# autoconf/configure.ac.
# Note that Makefile.config is covered by its own separate rule
# in Makefile.rules where it can be reused by sub-projects.
#------------------------------------------------------------------------
FilesToConfig := \
  bindings/ocaml/llvm/META.llvm \
  docs/doxygen.cfg \
  llvm.spec \
  include/llvm/Config/config.h \
  include/llvm/Config/llvm-config.h \
  include/llvm/Config/Targets.def \
  include/llvm/Config/AsmPrinters.def \
  include/llvm/Config/AsmParsers.def \
  include/llvm/Config/Disassemblers.def \
  include/llvm/Support/DataTypes.h
FilesToConfigPATH  := $(addprefix $(LLVM_OBJ_ROOT)/,$(FilesToConfig))

all-local:: $(FilesToConfigPATH)
$(FilesToConfigPATH) : $(LLVM_OBJ_ROOT)/% : $(LLVM_SRC_ROOT)/%.in
	$(Echo) Regenerating $*
	$(Verb) cd $(LLVM_OBJ_ROOT) && $(ConfigStatusScript) $*
.PRECIOUS: $(FilesToConfigPATH)

# NOTE: This needs to remain as the last target definition in this file so
# that it gets executed last.
ifneq ($(BUILD_DIRS_ONLY),1)
all::
	$(Echo) '*****' Completed $(BuildMode) Build
ifneq ($(ENABLE_OPTIMIZED),1)
	$(Echo) '*****' Note: Debug build can be 10 times slower than an
	$(Echo) '*****' optimized build. Use 'make ENABLE_OPTIMIZED=1' to
	$(Echo) '*****' make an optimized build. Alternatively you can
	$(Echo) '*****' configure with --enable-optimized.
ifeq ($(SHOW_DIAGNOSTICS),1)
	$(Verb) if test -s $(LLVM_OBJ_ROOT)/$(BuildMode)/diags; then \
	  $(LLVM_SRC_ROOT)/utils/clang-parse-diagnostics-file -a \
	    $(LLVM_OBJ_ROOT)/$(BuildMode)/diags; \
	fi
endif
endif
endif

check-llvm2cpp:
	$(Verb)$(MAKE) check TESTSUITE=Feature RUNLLVM2CPP=1

srpm: $(LLVM_OBJ_ROOT)/llvm.spec
	rpmbuild -bs $(LLVM_OBJ_ROOT)/llvm.spec

rpm: $(LLVM_OBJ_ROOT)/llvm.spec
	rpmbuild -bb --target $(TARGET_TRIPLE) $(LLVM_OBJ_ROOT)/llvm.spec

show-footprint:
	$(Verb) du -sk $(LibDir)
	$(Verb) du -sk $(ToolDir)
	$(Verb) du -sk $(ExmplDir)
	$(Verb) du -sk $(ObjDir)

build-for-llvm-top:
	$(Verb) if test ! -f ./config.status ; then \
	  ./configure --prefix="$(LLVM_TOP)/install" \
	    --with-llvm-gcc="$(LLVM_TOP)/llvm-gcc" ; \
	fi
	$(Verb) $(MAKE) tools-only

SVN = svn
SVN-UPDATE-OPTIONS =
AWK = awk

# Multiline variable defining a recursive function for finding svn repos rooted at
# a given path. svnup() requires one argument: the root to search from.
define SUB_SVN_DIRS
svnup() {
  dirs=`svn status --no-ignore $$1 | awk '/^(I|\?) / {print $$2}' | LC_ALL=C xargs svn info 2>/dev/null | awk '/^Path:\ / {print $$2}'`;
  if [ "$$dirs" = "" ]; then
    return;
  fi;
  for f in $$dirs; do
	  echo $$f;
    svnup $$f;
  done
}
endef
export SUB_SVN_DIRS

update:
	$(SVN) $(SVN-UPDATE-OPTIONS) update $(LLVM_SRC_ROOT)
	@eval $$SUB_SVN_DIRS; $(SVN) status --no-ignore $(LLVM_SRC_ROOT) | svnup $(LLVM_SRC_ROOT) | xargs $(SVN) $(SVN-UPDATE-OPTIONS) update

happiness: update all check-all

.PHONY: srpm rpm update happiness

# declare all targets at this level to be serial:

.NOTPARALLEL:

else # Building "Apple-style."
# In an Apple-style build, once configuration is done, lines marked
# "Apple-style" are removed with sed!  Please don't remove these!
# Look for the string "Apple-style" in utils/buildit/build_llvm.
include $(shell find . -name GNUmakefile) # Building "Apple-style."
endif # Building "Apple-style."
