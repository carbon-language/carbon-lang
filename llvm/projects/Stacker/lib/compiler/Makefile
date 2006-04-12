##===- projects/Stacker/lib/compiler/Makefile --------------*- Makefile -*-===##

LEVEL := ../..
LIBRARYNAME := stkr_compiler
EXTRA_DIST := Lexer.cpp.cvs Lexer.l.cvs \
	      StackerParser.cpp.cvs StackerParser.h.cvs StackerParser.y.cvs

include $(LEVEL)/Makefile.common

ifdef PARSE_DEBUG
INCLUDES += -DPARSE_DEBUG
endif

$(ObjDir)/Lexer.o : $(PROJ_SRC_DIR)/StackerParser.h
