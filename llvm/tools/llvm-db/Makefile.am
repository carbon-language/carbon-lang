#===-- tools/llvm-db/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = llvm-db

llvm_db_SOURCES = CLIDebugger.cpp Commands.cpp llvm-db.cpp

THEOBJS   = JIT CodeGen ExecutionEngine
THELIBS   =

# You can enable the X86 JIT on a non-X86 host by setting the flag
# ENABLE_X86_JIT on the make command line. If not, it will still be
# enabled automagically on an X86 host.
if ARCH_X86
  ENABLE_X86_JIT = 1
endif

# You can enable the Sparc JIT on a non-Sparc host by setting the flag
# ENABLE_SPARC_JIT on the make command line. If not, it will still be
# enabled automagically on an Sparc host.
if ARCH_SPARC
  ENABLE_SPARC_JIT = 1
endif

# What the X86 JIT requires
 ifdef ENABLE_X86_JIT
  CPPFLAGS += -DENABLE_X86_JIT
  THEOBJS  += X86 SelectionDAG
 endif

# What the Sparc JIT requires
 ifdef ENABLE_SPARC_JIT
#  CPPFLAGS += -DENABLE_SPARC_JIT
#  THEOBJS  += SparcV9 SparcV9InstrSched SparcV9LiveVar SparcV9RegAlloc
#  THEOBJS  += ProfilePaths BCWriter
#  THELIBS  += Instrument Transforms IPO IPA DataStructure
 endif

llvm_db_LDADD = \
  $(call GETOBJS,Interpreter,$(THEOBJS)) \
  $(call GETLIBS,$(THELIBS)) \
  $(call GETOBJS,ScalarOpts) \
  $(call GETLIBS,Analysis,TransformUtils,Target) \
  $(call GETOBJS,Debugger,BCReader,Core) \
  $(call GETLIBS,Support,Target,System)
