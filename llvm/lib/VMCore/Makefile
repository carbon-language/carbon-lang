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

BUILT_SOURCES = $(LEVEL)/include/llvm/intrinsics.gen

include $(LEVEL)/Makefile.config
ifeq ($(ARCH),Alpha)
BUILD_ARCHIVE = 1
endif

include $(LEVEL)/Makefile.common

$(LEVEL)/include/llvm/intrinsics.gen: $(LEVEL)/include/llvm/intrinsics.td
	@echo Building intrinsics.gen from intrinsics.td
	$(Verb) $(TableGen) $< -o $@ -gen-intrinsic

