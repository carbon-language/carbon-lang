##===- projects/Stacker/lib/compiler/Makefile --------------*- Makefile -*-===##

LEVEL := ../..
LIBRARYNAME := stkr_compiler

include $(LEVEL)/Makefile.common

ifdef PARSE_DEBUG
INCLUDES += -DPARSE_DEBUG
endif

$(ObjDir)/Lexer.o : $(PROJ_SRC_DIR)/StackerParser.h
