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

$(GENFILE): $(PROJ_SRC_ROOT)/include/llvm/Intrinsics.td $(TBLGEN)
	$(Echo) Building Intrinsics.gen from Intrinsics.td
	$(Verb) $(TableGen) $< -o $@ -gen-intrinsic

install-local:: $(GENFILE)
	$(DataInstall) $(GENFILE) $(PROJ_includedir)/llvm/Intrinsics.gen
