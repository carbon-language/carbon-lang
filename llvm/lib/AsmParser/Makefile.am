#===-- lib/AsmParser/Makefile.am ---------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

libexec_PROGRAMS = LLVMAsmParser.o

BUILT_SOURCES = llvmAsmParser.cpp Lexer.cpp llvmAsmParser.h

LLVMAsmParser_o_SOURCES = Parser.cpp $(BUILT_SOURCES)
LIBS=

Lexer.o: llvmAsmParser.h
