##===- utils/llvm-lit/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

include $(LEVEL)/Makefile.common

all:: $(ToolDir)/llvm-lit

$(ToolDir)/llvm-lit: llvm-lit.in $(ToolDir)/.dir
	$(Echo) "Creating 'llvm-lit' script..."
	$(Verb)sed -e "s#@LLVM_SOURCE_DIR@#$(LLVM_SRC_ROOT)#g" \
	     -e "s#@LLVM_BINARY_DIR@#$(LLVM_OBJ_ROOT)#g" \
	     $< > $@
	$(Verb)chmod +x $@
