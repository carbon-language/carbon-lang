# defs.mk
# $Revision: 42951 $
# $Date: 2014-01-21 14:41:41 -0600 (Tue, 21 Jan 2014) $

#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# --------------------------------------------------------------------------------------------------
# This file contains definitions common for OpenMP RTL and DSL makefiles.
# --------------------------------------------------------------------------------------------------

# Include really common definitions.
include $(LIBOMP_WORK)tools/src/common-defs.mk

#
# Directories.
#

# Check and normalize LIBOMP_EXPORTS.
ifeq "$(LIBOMP_EXPORTS)" ""
    $(error LIBOMP_EXPORTS environment variable must be set)
endif
ifneq "$(words $(LIBOMP_EXPORTS))" "1"
    $(error LIBOMP_EXPORTS must not contain spaces)
endif
override LIBOMP_EXPORTS := $(subst \,/,$(LIBOMP_EXPORTS))
ifeq "$(filter %/,$(LIBOMP_EXPORTS))" ""
    override LIBOMP_EXPORTS := $(LIBOMP_EXPORTS)/
endif
# Output directories.
out_dir      = $(LIBOMP_EXPORTS)
out_cmn_dir  = $(out_dir)common$(suffix)/
out_ptf_dir  = $(out_dir)$(platform)$(suffix)/
_out_lib_dir = $(out_dir)$(1)$(suffix)/lib$(if $(filter mac_%,$(1)),.thin)/
out_lib_dir  = $(call _out_lib_dir,$(platform))
ifneq "$(arch)" "mic"
out_l10n_dir = $(out_lib_dir)$(if $(filter lin mac,$(os)),locale/)
else
out_l10n_dir = $(out_lib_dir)
endif
ifeq "$(os)" "mac"
    _out_lib_fat_dir = $(out_dir)$(1)$(suffix)/lib/
    out_lib_fat_dir  = $(call _out_lib_fat_dir,$(platform))
    out_l10n_fat_dir = $(out_lib_fat_dir)locale/
endif

#
# Retrieve build number,
#

ifeq "$(clean)" ""
    # Parse kmp_version.c file, look for "#define KMP_VERSION_BUILD yyyymmdd" string,
    # leave only "yyyymmdd". Note: Space after $$1 is important, it helps to detect possible errors.
    build := $(strip $(shell $(perl) -p -e '$$_ =~ s{^(?:\s*\#define\s+KMP_VERSION_BUILD\s+([0-9]{8})|.*)\s*\n}{$$1 }' $(LIBOMP_WORK)src/kmp_version.c))
    ifneq "$(words $(build))" "1"
        $(error Failed to pase "kmp_version.c", cannot extract build number)
    endif
    $(call say,Build  : $(build)$(if $(filter 00000000,$(build)), (development)))
endif

# end of file #
