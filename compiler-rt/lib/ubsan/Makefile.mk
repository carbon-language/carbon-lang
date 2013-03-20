#===- lib/ubsan/Makefile.mk ---------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := ubsan
SubDirs :=

Sources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
CXXSources := ubsan_type_hash.cc ubsan_handlers_cxx.cc
CSources := $(filter-out $(CXXSources),$(Sources))
ObjNames := $(Sources:%.cc=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../sanitizer_common/*.h)

# Define a convenience variable for all the ubsan functions.
UbsanFunctions := $(CSources:%.cc=%)
UbsanCXXFunctions := $(CXXSources:%.cc=%)
