##===- tools/llvm-nm/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = llvm-nm
LINK_COMPONENTS = archive bitreader

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1
CXXFLAGS = -fno-rtti

include $(LEVEL)/Makefile.common
