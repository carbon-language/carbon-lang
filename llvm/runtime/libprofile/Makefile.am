#===-- runtime/libprofile/Makefile.am ----------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

lib_LIBRARIES = libprofile.a
libprofile_a_SOURCES = BasicBlockTracing.c BlockProfiling.c CommonProfiling.c \
                       EdgeProfiling.c FunctionProfiling.c Profiling.h

EXPORTED_SYMBOL_LIST = exported_symbols.lst

include $(top_srcdir)/Makefile_config
