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

.ver:
	@if [ '$(SVN_REVISION)' != '$(shell cat .ver-svn 2>/dev/null)' ]; then	\
		echo '$(SVN_REVISION)' > .ver-svn;			\
	fi
.ver-svn: .ver
$(ObjDir)/Driver.o: .ver-svn
