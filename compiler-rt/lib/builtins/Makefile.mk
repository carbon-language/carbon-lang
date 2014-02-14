#===- lib/builtins/Makefile.mk -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

ModuleName := builtins
SubDirs :=

# Add arch specific optimized implementations.
SubDirs += i386 ppc x86_64 arm

# Define the variables for this specific directory.
Sources := $(foreach file,$(wildcard $(Dir)/*.c),$(notdir $(file)))
ObjNames := $(Sources:%.c=%.o)
Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
