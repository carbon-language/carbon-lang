##===- lib/AsmParser/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME := LLVMAsmParser
BUILT_SOURCES := llvmAsmParser.cpp llvmAsmParser.h Lexer.cpp

include $(LEVEL)/Makefile.common

#
# Make the object code file for the lexer depend upon the header file generated
# by the Bison parser.  This prevents the Lexer from being compiled before the
# header file it needs is built.
$(OBJDIR)/Lexer.o: llvmAsmParser.h
