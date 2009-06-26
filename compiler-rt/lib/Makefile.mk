#===- lib/Makefile.mk --------------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

Dir := lib
SubDirs := i386 ppc x86_64

Sources := $(foreach file,$(wildcard $(Dir)/*.c),$(notdir $(file)))
ObjNames := $(Sources:%.c=%.o)
Target := Generic

# FIXME: use automatic dependencies?
Dependencies := $(wildcard $(Dir)/*.h)

include make/subdir.mk
