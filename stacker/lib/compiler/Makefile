##===- projects/Stacker/lib/compiler/Makefile --------------*- Makefile -*-===##

LEVEL := ../..
LIBRARYNAME := stkr_compiler
EXTRA_DIST := Lexer.cpp.cvs Lexer.l.cvs \
	      StackerParser.cpp.cvs StackerParser.h.cvs StackerParser.y.cvs
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

ifdef PARSE_DEBUG
INCLUDES += -DPARSE_DEBUG
endif

# Disable -pedantic for this library, as bison output isn't necessarily 
# -pedantic clean.
CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))

$(ObjDir)/Lexer.o : $(PROJ_SRC_DIR)/StackerParser.h
$(ObjDir)/StackerCompiler.o : $(PROJ_SRC_DIR)/StackerParser.h
