##===- lib/AsmParser/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME := LLVMAsmParser
BUILD_ARCHIVE = 1
EXTRA_DIST := llvmAsmParser.cpp.cvs llvmAsmParser.h.cvs llvmAsmParser.y.cvs

include $(LEVEL)/Makefile.common

# Disable -pedantic for this library, as bison output isn't necessarily 
# -pedantic clean.
CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))

#
# Make the object code file for the lexer depend upon the header file generated
# by the Bison parser.  This prevents the Lexer from being compiled before the
# header file it needs is built.
$(ObjDir)/LLLexer.o: $(PROJ_SRC_DIR)/llvmAsmParser.h
