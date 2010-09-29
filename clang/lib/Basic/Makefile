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

CLANG_LEVEL := ../..
LIBRARYNAME := clangBasic

include $(CLANG_LEVEL)/Makefile

SVN_REVISION := $(strip \
	$(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(PROJ_SRC_DIR)/../..))

SVN_REPOSITORY := $(strip \
	$(shell $(LLVM_SRC_ROOT)/utils/GetRepositoryPath $(PROJ_SRC_DIR)/../..))

CPP.Defines += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
         -DSVN_REVISION='"$(SVN_REVISION)"' -DSVN_REPOSITORY='"$(SVN_REPOSITORY)"'

$(ObjDir)/.ver-svn .ver: $(ObjDir)/.dir
	@if [ '$(SVN_REVISION)' != '$(shell cat $(ObjDir)/.ver-svn 2>/dev/null)' ]; then\
		echo '$(SVN_REVISION)' > $(ObjDir)/.ver-svn;			\
	fi
$(ObjDir)/.ver-svn: .ver
$(ObjDir)/Version.o: $(ObjDir)/.ver-svn
