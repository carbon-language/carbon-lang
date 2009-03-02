##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..

TOOLNAME = clang-driver
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include
CXXFLAGS = -fno-rtti

LINK_COMPONENTS := system
USEDLIBS = clangDriver.a

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.common
