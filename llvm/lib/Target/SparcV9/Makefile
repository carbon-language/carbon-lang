##===- lib/Target/SparcV9/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMSparcV9
PARALLEL_DIRS = InstrSched LiveVar ModuloScheduling RegAlloc

TARGET = SparcV9

BUILT_SOURCES = \
  SparcV9GenCodeEmitter.inc \
  SparcV9.burm.cpp 

include $(LEVEL)/Makefile.common

SparcV9.burg.in1 : $(PROJ_SRC_DIR)/SparcV9.burg.in
	$(Echo) Pre-processing SparcV9.burg.in
	$(Verb) $(CXX) -E $(CPP.Flags) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/Ydefine/#define/' > $@

SparcV9.burm : SparcV9.burg.in1
	$(Echo) Pre-processing SparcV9.burg.in
	$(Verb) $(CXX) -E $(CPP.Flags) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/^Xinclude/#include/' | $(SED) 's/^Xdefine/#define/' > $@

SparcV9.burm.cpp: SparcV9.burm
	$(Echo) "Burging `basename $<`"
	$(Verb) $(BURG) -I $< -o $@

clean::
	$(Verb) $(RM) -f SparcV9.burg.in1 SparcV9.burm SparcV9.burm.cpp

