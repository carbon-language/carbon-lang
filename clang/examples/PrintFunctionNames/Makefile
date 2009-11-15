##===- examples/PrintFunctionNames/Makefile ----------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME = PrintFunctionNames

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

LINK_COMPONENTS := MC support
USEDLIBS = clangFrontend.a clangIndex.a clangSema.a clangAnalysis.a clangAST.a clangParse.a clangLex.a clangBasic.a

include $(LEVEL)/Makefile.common
