#===- lib/sanitizer_common/Makefile.mk ---------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := sanitizer_common
SubDirs :=

Sources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
NolibcSources := $(foreach file,$(wildcard $(Dir)/*_nolibc.cc),$(notdir $(file)))
Sources := $(filter-out $(NolibcSources),$(Sources))
ObjNames := $(Sources:%.cc=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)

# Define a convenience variable for all the sanitizer_common functions.
SanitizerCommonFunctions := $(Sources:%.cc=%)
