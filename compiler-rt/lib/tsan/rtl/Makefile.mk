#===- lib/tsan/rtl/Makefile.mk -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := tsan
SubDirs :=

Sources := $(foreach file,$(wildcard $(Dir)/*.cc),$(notdir $(file)))
AsmSources := $(foreach file,$(wildcard $(Dir)/*.S),$(notdir $(file)))
ObjNames := $(Sources:%.cc=%.o) $(AsmSources:%.S=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
Dependencies += $(wildcard $(Dir)/../../interception/*.h)
Dependencies += $(wildcard $(Dir)/../../interception/mach_override/*.h)

# Define a convenience variable for all the tsan functions.
TsanFunctions += $(Sources:%.cc=%) $(AsmSources:%.S=%)
