##===- clang/lib/Driver/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME := clangDriver
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

SVN_REVISION = $(shell cd $(PROJ_SRC_DIR)/../.. && svnversion)

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include -DTARGET_TRIPLE='"$(TARGET_TRIPLE)"' -DSVN_REVISION='"$(SVN_REVISION)"'

include $(LEVEL)/Makefile.common

$(ObjDir)/.ver-svn .ver:
	@if [ '$(SVN_REVISION)' != '$(shell cat $(ObjDir)/.ver-svn 2>/dev/null)' ]; then\
		echo '$(SVN_REVISION)' > $(ObjDir)/.ver-svn;			\
	fi
$(ObjDir)/.ver-svn: .ver
$(ObjDir)/Driver.o: $(ObjDir)/.ver-svn
