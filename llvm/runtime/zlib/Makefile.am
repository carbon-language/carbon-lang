#===-- runtime/zlib/Makefile.am ----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

lib_LIBRARIES = libz.a
libz_a_SOURCES = adler32.c compress.c crc32.c crc32.h deflate.c deflate.h \
                gzio.c infback.c inffast.c inffast.h inffixed.h \
                inflate.c inflate.h inftrees.c inftrees.h trees.c \
                trees.h uncompr.c zconf.h zlib.h zutil.c zutil.h

CPPFLAGS = -DNO_DUMMY_DECL=1

BYTECODE_LIBRARY = $(lib_LIBRARIES)

include $(top_srcdir)/Makefile_config
