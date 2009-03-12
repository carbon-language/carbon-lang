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

# FIXME: It is unfortunate we need to pull in the bitcode reader and
# writer just to get the serializer stuff used by clangBasic.
LINK_COMPONENTS := system support bitreader bitwriter

# FIXME: We shouldn't need clangLex.a here; we do because the
# TextDiagnosticPrinter is pulling it in. :(
USEDLIBS = clangDriver.a clangFrontend.a clangLex.a clangBasic.a

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.common
