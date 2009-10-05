##===- clang/lib/Basic/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the Basic library for the C-Language front-end.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME := clangBasic
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

# Don't install Clang libraries
NO_INSTALL = 1

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include

include $(LEVEL)/Makefile.common

SVN_REVISION := $(shell cd $(PROJ_SRC_DIR)/../.. && svnversion)

CPP.Defines += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
         -DSVN_REVISION='"$(SVN_REVISION)"'

$(ObjDir)/.ver-svn .ver: $(ObjDir)/.dir
	@if [ '$(SVN_REVISION)' != '$(shell cat $(ObjDir)/.ver-svn 2>/dev/null)' ]; then\
		echo '$(SVN_REVISION)' > $(ObjDir)/.ver-svn;			\
	fi
$(ObjDir)/.ver-svn: .ver
$(ObjDir)/Version.o: $(ObjDir)/.ver-svn
