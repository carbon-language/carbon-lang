##===- lib/Target/AMDGPU/Utils/Makefile --------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
LIBRARYNAME = LLVMAMDGPUUtils

# Hack: we need to include 'main' AMDGPU target directory to grab private
# headers
CPP.Flags += -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LEVEL)/Makefile.common
