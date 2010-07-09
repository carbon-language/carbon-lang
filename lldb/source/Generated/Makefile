##===- source/Generated/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbGenerated
BUILD_ARCHIVE = 1

BUILT_SOURCES = LLDB_vers.c

include $(LLDB_LEVEL)/Makefile

LLDB_vers.c: $(PROJ_SRC_DIR)/../../scripts/generate-vers.pl $(PROJ_SRC_DIR)/../../lldb.xcodeproj/project.pbxproj
	$(PROJ_SRC_DIR)/../../scripts/generate-vers.pl $(PROJ_SRC_DIR)/../../lldb.xcodeproj/project.pbxproj > $(PROJ_OBJ_DIR)/LLDB_vers.c
