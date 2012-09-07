#===- lib/asan/dynamic/Makefile.mk -------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := asan_dynamic
SubDirs :=

Sources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
ObjNames := $(Sources:%.cc=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../../interception/*.h)
Dependencies += $(wildcard $(Dir)/../../interception/mach_override/*.h)
Dependencies += $(wildcard $(Dir)/../../sanitizer_common/*.h)

# Define a convenience variable for the asan dynamic functions.
AsanDynamicFunctions := $(Sources:%.cc=%)
