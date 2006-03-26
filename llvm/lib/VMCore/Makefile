##===- lib/VMCore/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = LLVMCore

BUILT_SOURCES = $(PROJ_SRC_ROOT)/include/llvm/Intrinsics.gen

include $(LEVEL)/Makefile.config
ifeq ($(ARCH),Alpha)
BUILD_ARCHIVE = 1
endif

include $(LEVEL)/Makefile.common

GENFILE:=$(PROJ_SRC_ROOT)/include/llvm/Intrinsics.gen

INTRINSICTD  := $(PROJ_SRC_ROOT)/include/llvm/Intrinsics.td
INTRINSICTDS := $(wildcard $(PROJ_SRC_ROOT)/include/llvm/Intrinsics*.td)

$(GENFILE): $(INTRINSICTDS) $(TBLGEN)
	$(Echo) Building Intrinsics.gen from Intrinsics.td
	$(Verb) $(TableGen) $(INTRINSICTD) -o $@ -gen-intrinsic

install-local:: $(GENFILE)
	$(DataInstall) $(GENFILE) $(PROJ_includedir)/llvm/Intrinsics.gen
