##===- tools/llvm-db/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llvm-db

# Get the $(ARCH) setting
include $(LEVEL)/Makefile.config

# Generic JIT libraries
JITLIBS   = lli-jit codegen executionengine
ARCHLIBS  =

# You can enable the X86 JIT on a non-X86 host by setting the flag
# ENABLE_X86_JIT on the make command line. If not, it will still be
# enabled automagically on an X86 host.
ifeq ($(ARCH), x86)
  ENABLE_X86_JIT = 1
endif

# What the X86 JIT requires
ifdef ENABLE_X86_JIT
  CPPFLAGS += -DENABLE_X86_JIT
  JITLIBS  += x86 selectiondag
  # X86 doesn't require any ARCHLIBS
endif

# You can enable the Sparc JIT on a non-Sparc host by setting the flag
# ENABLE_SPARC_JIT on the make command line. If not, it will still be
# enabled automagically on an Sparc host.
ifeq ($(ARCH), Sparc)
  ENABLE_SPARC_JIT = 1
endif

# What the Sparc JIT requires
ifdef ENABLE_SPARC_JIT
  CPPFLAGS += -DENABLE_SPARC_JIT
  JITLIBS  += sparc 
  ARCHLIBS += sched livevar instrument.a profpaths \
              bcwriter transforms.a ipo.a ipa.a datastructure.a regalloc \
              select
endif

USEDLIBS = lli-interpreter $(JITLIBS) $(ARCHLIBS) scalaropts analysis.a \
           transformutils.a debugger bcreader vmcore support target.a


# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common

