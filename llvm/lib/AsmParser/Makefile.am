#===-- lib/AsmParser/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMAsmParser.a

BUILT_SOURCES = llvmAsmParser.cpp Lexer.cpp llvmAsmParser.h

libLLVMAsmParser_a_SOURCES = Parser.cpp $(BUILT_SOURCES)

PRELINK=libLLVMAsmParser.a

Lexer.o: llvmAsmParser.h
