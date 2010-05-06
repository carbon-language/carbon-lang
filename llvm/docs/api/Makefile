##===- docs/api/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

include $(LEVEL)/Makefile.common

API.Project  = LLVM
API.project  = llvm
API.srcdir   = $(PROJ_SRC_ROOT)

include $(LLVM_SRC_ROOT)/docs/mk/api.mk

# Define API.skeleton=1 to generate only a subset of API; for testing purposes.
#
ifneq (undefined,$(origin API.skeleton))
$(API.doxygen.target): API.doxygen.cfg.opts += INPUT=$(PROJ_SRC_DIR)/intro.h
$(API.doxygen.target): API.doxygen.cfg.opts += INPUT+=$(PROJ_SRC_ROOT)/include/llvm/Target
endif
