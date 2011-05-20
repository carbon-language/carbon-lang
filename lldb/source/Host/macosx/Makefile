##===- source/Host/macosx/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../..
LIBRARYNAME := lldbHostMacOSX
BUILD_ARCHIVE = 1

SOURCES = $(notdir $(wildcard $(PROJ_SRC_DIR)/*.cpp $(PROJ_SRC_DIR)/*.mm)) \
          $(addprefix cfcpp/,$(notdir $(wildcard $(PROJ_SRC_DIR)/cfcpp/*.cpp)))

# Make sure the cfcpp output directory exists
all-local:: $(PROJ_OBJ_DIR)/$(BuildMode)/cfcpp

$(PROJ_OBJ_DIR)/$(BuildMode)/cfcpp:
	-$(Verb) $(MKDIR) $(PROJ_OBJ_DIR)/$(BuildMode)/cfcpp 2>/dev/null

include $(LLDB_LEVEL)/Makefile

