#===-- runtime/libpng/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

lib_LIBRARIES = libpng.a
libpng_a_SOURCES = example.c png.c pngerror.c pnggccrd.c pngget.c pngmem.c \
                   pngpread.c pngread.c pngrio.c pngrtran.c pngrutil.c \
                   pngset.c pngtest.c pngtrans.c pngvcrd.c pngwio.c pngwrite.c \
                   pngwtran.c pngwutil.c

CPPFLAGS = -I../zlib

BYTECODE_LIBRARY = libpng.a

include $(top_srcdir)/Makefile_config
