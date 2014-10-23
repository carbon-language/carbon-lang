##===- tools/llvm-go/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
include $(LEVEL)/Makefile.common

all:: $(ToolDir)/llvm-go$(EXEEXT)

$(ToolDir)/llvm-go$(EXEEXT): $(PROJ_SRC_DIR)/llvm-go.go
	$(GO) build -o $@ $<
