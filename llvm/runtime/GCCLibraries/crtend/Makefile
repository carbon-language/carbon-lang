##===- runtime/GCCLibraries/crtend/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
BYTECODE_LIBRARY=1
DONT_BUILD_RELINKED=1
LIBRARYNAME=crtend

Source = $(notdir $(wildcard $(SourceDir)/*.cpp $(SourceDir)/*.c)) listend.ll

EXPORTED_SYMBOL_FILE = $(SourceDir)/exported_symbol_list.lst

include $(LEVEL)/Makefile.common


$(LLVMGCCDIR)/bytecode-libs/crtend.o: $(LIBNAME_BC)
	@cp $< $@

install:: $(LLVMGCCDIR)/bytecode-libs/crtend.o
	@rm $(LLVMGCCDIR)/bytecode-libs/libcrtend.bc

