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

include $(LEVEL)/Makefile.common

#
# Make the source file depend on the header file.  In this way, dependencies
# (which depend on the source file) won't get generated until bison is done
# generating the C source and header files for the parser.
#
$(ObjDir)/FileLexer.o : $(PROJ_SRC_DIR)/FileParser.h 
