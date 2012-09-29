#===- lib/arm/Makefile.mk ----------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := builtins
SubDirs := 
OnlyArchs := armv5 armv6 armv7 armv7f armv7k armv7s

AsmSources := $(foreach file,$(wildcard $(Dir)/*.S),$(notdir $(file)))
Sources := $(foreach file,$(wildcard $(Dir)/*.c),$(notdir $(file)))
ObjNames := $(Sources:%.c=%.o) $(AsmSources:%.S=%.o)
Implementation := Optimized

# FIXME: use automatic dependencies?
Dependencies := $(wildcard lib/*.h $(Dir)/*.h)
