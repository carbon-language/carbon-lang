#===- lib/asan/Makefile.mk ---------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := asan
SubDirs := 

CCSources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
SSources := $(foreach file,$(wildcard $(Dir)/*.S),$(notdir $(file)))
Sources := $(CCSources) $(SSources)
ObjNames := $(CCSources:%.cc=%.o) $(SSources:%.S=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../interception/*.h)
Dependencies += $(wildcard $(Dir)/../sanitizer_common/*.h)

# Define a convenience variable for all the asan functions.
AsanFunctions := $(CCSources:%.cc=%) $(SSources:%.S=%)
