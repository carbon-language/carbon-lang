##===- tools/index-test/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..

TOOLNAME = index-test
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti
NO_INSTALL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.config

LINK_COMPONENTS := bitreader mc
USEDLIBS = clangIndex.a clangFrontend.a clangDriver.a clangAnalysis.a clangSema.a \
	   clangAST.a clangParse.a clangLex.a clangBasic.a

include $(LLVM_SRC_ROOT)/Makefile.rules
