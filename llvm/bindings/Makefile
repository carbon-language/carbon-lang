##===- bindings/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ..

include $(LEVEL)/Makefile.config

PARALLEL_DIRS =

ifneq (,$(filter ocaml,$(BINDINGS_TO_BUILD)))
PARALLEL_DIRS += ocaml
endif

include $(LEVEL)/Makefile.common
