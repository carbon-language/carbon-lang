#===- lib/interception/mach_override/Makefile.mk -----------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := interception
SubDirs :=

Sources := $(foreach file,$(wildcard $(Dir)/*.c),$(notdir $(file)))
ObjNames := $(Sources:%.c=%.o)

Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)

# Define a convenience variable for all the interception functions.
InterceptionFunctions += $(Sources:%.c=%)
