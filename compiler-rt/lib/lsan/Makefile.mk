#===- lib/lsan/Makefile.mk ---------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := lsan
SubDirs := 

Sources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
ObjNames := $(Sources:%.cc=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../interception/*.h)
Dependencies += $(wildcard $(Dir)/../sanitizer_common/*.h)

# Define a convenience variable for all the lsan functions.
LsanFunctions := $(Sources:%.cc=%)

# lsan functions used in another sanitizers.
LsanCommonSources := $(foreach file,$(wildcard $(Dir)/lsan_common*.cc),$(notdir $(file)))
LsanCommonFunctions := $(LsanCommonSources:%.cc=%)
