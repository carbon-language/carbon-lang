##===- lib/LibDriver/Makefile ------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMLibDriver
BUILD_ARCHIVE := 1
BUILT_SOURCES = Options.inc
TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

$(ObjDir)/Options.inc.tmp : Options.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building lib Driver Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<
