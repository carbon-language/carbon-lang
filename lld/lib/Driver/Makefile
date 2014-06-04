##===- lld/lib/Driver/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLD_LEVEL := ../..
LIBRARYNAME := lldDriver

BUILT_SOURCES = CoreOptions.inc UniversalDriverOptions.inc DarwinLdOptions.inc \
                GnuLdOptions.inc WinLinkOptions.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LLD_LEVEL)/Makefile

$(ObjDir)/CoreOptions.inc.tmp : CoreOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building LLD CoreOptions Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<

$(ObjDir)/UniversalDriverOptions.inc.tmp : UniversalDriverOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building LLD Universal Driver Options tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<

$(ObjDir)/DarwinLdOptions.inc.tmp : DarwinLdOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building LLD Darwin ld Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<

$(ObjDir)/GnuLdOptions.inc.tmp : GnuLdOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building LLD Gnu ld Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<

$(ObjDir)/WinLinkOptions.inc.tmp : WinLinkOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building LLD WinLinkOptions Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<
