#===-- runtime/GC/SemiSpace/Makefile.am --------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

lib_LIBRARIES = libGCSemiSpace.a
libGCSemiSpace_a_SOURCES = semispace.c

BYTECODE_LIBRARY = $(lib_LIBRARIES)

EXPORTED_SYMBOL_FILE = $(srcdir)/../gc_exported_symbols.lst

include $(top_srcdir)/Makefile_config
