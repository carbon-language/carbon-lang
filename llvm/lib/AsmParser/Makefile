
LEVEL = ../..

LIBRARYNAME = asmparser

include $(LEVEL)/Makefile.common

#
# Make the source code file for the lexer depend upon the header file generated
# by the Bison parser.  This prevents the generation of dependencies from
# being performed until after the header file has been created.
#
Lexer.cpp: llvmAsmParser.h
