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

CLANG_REVISION := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(PROJ_SRC_DIR)/../..))

CLANG_REPOSITORY := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetRepositoryPath $(PROJ_SRC_DIR)/../..))

LLVM_REVISION := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(LLVM_SRC_ROOT)))

LLVM_REPOSITORY := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetRepositoryPath $(LLVM_SRC_ROOT)))

CPP.Defines += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
         -DCLANG_REVISION='"$(CLANG_REVISION)"' -DCLANG_REPOSITORY='"$(CLANG_REPOSITORY)"' \
         -DLLVM_REVISION='"$(LLVM_REVISION)"' -DLLVM_REPOSITORY='"$(LLVM_REPOSITORY)"'

$(ObjDir)/.ver-svn .ver: $(ObjDir)/.dir
	@if [ '$(CLANG_REVISION) $(LLVM_REVISION)' != '$(shell cat $(ObjDir)/.ver-svn 2>/dev/null)' ]; then\
		echo '$(CLANG_REVISION) $(LLVM_REVISION)' > $(ObjDir)/.ver-svn;                    \
	fi
$(ObjDir)/.ver-svn: .ver
$(ObjDir)/Version.o: $(ObjDir)/.ver-svn
