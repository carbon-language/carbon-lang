##===- projects/Stacker/lib/compiler/Makefile --------------*- Makefile -*-===##

LEVEL := ../..
LIBRARYNAME := stkr_compiler
BUILT_SOURCES := StackerParser.cpp StackerParser.h Lexer.cpp

include $(LEVEL)/Makefile.common

ifdef PARSE_DEBUG
INCLUDES += -DPARSE_DEBUG
endif

$(OBJDIR)/Lexer.o : StackerParser.h
