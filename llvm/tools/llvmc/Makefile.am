#===-- tools/llvmc/Makefile.am -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

bin_PROGRAMS = llvmc

BUILT_SOURCES = ConfigLexer.cpp

llvmc_SOURCES = CompilerDriver.cpp Configuration.cpp llvmc.cpp $(BUILT_SOURCES)

#dist_EXTRA_llvmc_SOURCES = ConfigLexer.l

llvmc_LDADD = $(call GETOBJS,BCReader,Core) \
              $(call GETLIBS,Support,System)
