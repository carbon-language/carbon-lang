#===-- lib/ExecutionEngine/JIT/Makefile.am -----------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

libexec_PROGRAMS = LLVMJIT.o

LLVMJIT_o_SOURCES = Emitter.cpp Intercept.cpp JIT.cpp TargetSelect.cpp
LIBS=

# Enable the X86 JIT if compiling on X86
if ARCH_X86
  ENABLE_X86_JIT = 1
endif

# This flag can also be used on the command line to force inclusion
# of the X86 JIT on non-X86 hosts
 ifdef ENABLE_X86_JIT
  CPPFLAGS += -DENABLE_X86_JIT
 endif

# Enable the Sparc JIT if compiling on Sparc
if ARCH_SPARC
  ENABLE_SPARC_JIT = 1
endif

# This flag can also be used on the command line to force inclusion
# of the Sparc JIT on non-Sparc hosts
 ifdef ENABLE_SPARC_JIT
  CPPFLAGS += -DENABLE_SPARC_JIT
 endif
