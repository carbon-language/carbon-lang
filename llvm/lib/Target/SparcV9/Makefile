##===- lib/Target/SparcV9/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = sparcv9
PARALLEL_DIRS = InstrSched LiveVar ModuloScheduling RegAlloc

BUILT_SOURCES = \
  SparcV9CodeEmitter.inc \
  SparcV9.burm.cpp 

include $(LEVEL)/Makefile.common

SparcV9.burg.in1 : $(BUILD_SRC_DIR)/SparcV9.burg.in
	$(CXX) -E $(CPPFLAGS) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/Ydefine/#define/' > $@

SparcV9.burm : SparcV9.burg.in1
	$(CXX) -E $(CPPFLAGS) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/^Xinclude/#include/' | $(SED) 's/^Xdefine/#define/' > $@

SparcV9.burm.cpp: SparcV9.burm
	@echo "Burging `basename $<`"
	$(BURG) -I $< -o $@

TABLEGEN_FILES := $(notdir $(wildcard $(BUILD_SRC_DIR)/*.td))

SparcV9CodeEmitter.inc: $(BUILD_SRC_DIR)/SparcV9.td $(TABLEGEN_FILES) $(TBLGEN)
	@echo "Running tblgen on SparcV9.td"
	$(TableGen) -gen-emitter -o $@ $<

clean::
	$(VERB) $(RM) -f SparcV9CodeEmitter.inc SparcV9.burg.in1 SparcV9.burm SparcV9.burm.cpp

