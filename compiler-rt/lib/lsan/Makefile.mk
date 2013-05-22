#===- lib/lsan/Makefile.mk ---------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := lsan_common
SubDirs := 

Sources := $(foreach file,$(wildcard $(Dir)/lsan_common*.cc),$(notdir $(file)))
ObjNames := $(Sources:%.cc=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../sanitizer_common/*.h)

# Define a convenience variable for all the asan functions.
LsanCommonFunctions := $(Sources:%.cc=%)
