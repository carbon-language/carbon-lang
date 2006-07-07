##===- utils/TableGen/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = tblgen
USEDLIBS = LLVMSupport.a LLVMSystem.a
EXTRA_DIST = FileLexer.cpp.cvs FileLexer.l.cvs \
	     FileParser.cpp.cvs FileParser.h.cvs FileParser.y.cvs
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

# Disable -pedantic for tblgen
CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))

#
# Make the source file depend on the header file.  In this way, dependencies
# (which depend on the source file) won't get generated until bison is done
# generating the C source and header files for the parser.
#
$(ObjDir)/FileLexer.o : $(PROJ_SRC_DIR)/FileParser.h 
