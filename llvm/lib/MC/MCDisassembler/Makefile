##===- lib/MC/MCDisassembler/Makefile ----------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMMCDisassembler
BUILT_SOURCES = EDInfo.inc

include $(LEVEL)/Makefile.common

$(ObjDir)/EDInfo.inc:	$(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building semantic information header"
	$(Verb) $(TableGen) -o $@ -gen-enhanced-disassembly-header /dev/null

clean::
	-$(Verb) $(RM) -f EDInfo.inc
