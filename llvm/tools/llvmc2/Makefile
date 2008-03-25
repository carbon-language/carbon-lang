##===- tools/llvmcc/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = llvmc2
BUILT_SOURCES = Tools.inc
LINK_COMPONENTS = support system
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

TOOLS_TARGET=default
ifeq ($(TOOLS_TARGET), default)
	TOOLS_SOURCE=Example.td
else
	TOOLS_SOURCE=ExampleWithOpt.td
endif

# TOFIX: integrate this part into Makefile.rules?
# The degree of horrorshowness in that file is too much for me atm.
$(ObjDir)/Tools.inc.tmp: $(TOOLS_SOURCE) $(ObjDir)/.dir
	$(Echo) "Building LLVMCC configuration library with tblgen"
	$(Verb) $(TableGen) -gen-llvmcc -o $(call SYSPATH, $@) $<

Tools.inc : $(ObjDir)/Tools.inc.tmp
	$(Verb) $(CMP) -s $@ $< || $(CP) $< $@

