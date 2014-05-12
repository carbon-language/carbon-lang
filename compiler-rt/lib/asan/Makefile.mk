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
CXXOnlySources := asan_new_delete.cc
COnlySources := $(filter-out $(CXXOnlySources),$(CCSources))
SSources := $(foreach file,$(wildcard $(Dir)/*.S),$(notdir $(file)))
Sources := $(CCSources) $(SSources)
ObjNames := $(CCSources:%.cc=%.o) $(SSources:%.S=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../interception/*.h)
Dependencies += $(wildcard $(Dir)/../sanitizer_common/*.h)

# Define a convenience variable for all the asan functions.
AsanFunctions := $(COnlySources:%.cc=%) $(SSources:%.S=%)
AsanCXXFunctions := $(CXXOnlySources:%.cc=%)
