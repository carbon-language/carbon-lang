#===- lib/Makefile.mk --------------------------------------*- Makefile -*--===#
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

# Add other submodules.
SubDirs += asan
SubDirs += interception
SubDirs += profile
SubDirs += sanitizer_common
SubDirs += tsan

# FIXME: We don't currently support building an atomic library, and as it must
# be a separate library from the runtime library, we need to remove its source
# code from the source files list.
ExcludedSources := atomic.c

# Define the variables for this specific directory.
Sources := $(foreach file,$(wildcard $(Dir)/*.c),$(filter-out $(ExcludedSources),$(notdir $(file))))
ObjNames := $(Sources:%.c=%.o)
Implementation := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)
