#===-- lib/Target/SparcV9/Makefile.am ----------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

SUBDIRS = InstrSched LiveVar RegAlloc

libexec_PROGRAMS = LLVMSparcV9.o

BUILT_SOURCES = \
  SparcV9CodeEmitter.inc \
  SparcV9.burm.cpp 

LLVMSparcV9_o_SOURCES = \
  EmitBytecodeToAssembly.cpp \
  InternalGlobalMapper.cpp \
  MachineCodeForInstruction.cpp \
  MachineFunctionInfo.cpp \
  MappingInfo.cpp \
  SparcV9AsmPrinter.cpp \
  SparcV9BurgISel.cpp \
  SparcV9CodeEmitter.cpp \
  SparcV9FrameInfo.cpp \
  SparcV9PeepholeOpts.cpp \
  SparcV9PreSelection.cpp \
  SparcV9PrologEpilogInserter.cpp \
  SparcV9RegClassInfo.cpp \
  SparcV9RegInfo.cpp \
  SparcV9RegisterInfo.cpp \
  SparcV9SchedInfo.cpp \
  SparcV9StackSlots.cpp \
  SparcV9TargetMachine.cpp \
  SparcV9TmpInstr.cpp \
  $(BUILT_SOURCES)

LIBS=

SparcV9.burg.in1 : SparcV9.burg.in
	$(CXX) -E $(AM_CPPFLAGS) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/Ydefine/#define/' > $@

SparcV9.burm : SparcV9.burg.in1
	$(CXX) -E -I$(AM_CPPFLAGS) -x c++ $< | $(SED) '/^#/d' | $(SED) 's/^Xinclude/#include/' | $(SED) 's/^Xdefine/#define/' > $@

SparcV9.burm.cpp: SparcV9.burm
	@echo "Burging `basename $<`"
	$(RunBurg) $< -o $@

SparcV9CodeEmitter.inc: SparcV9.td $(TABLEGEN_FILES) $(TBLGEN)
	@echo "Tblgen'ing `basename $<`"
	$(TBLGEN) -I $(srcdir) $< -gen-emitter -o $@

TABLEGEN_FILES := $(notdir $(wildcard $(SourceDir)/*.td))

clean::
	$(VERB) $(RM) -f SparcV9CodeEmitter.inc SparcV9.burg.in1 SparcV9.burm SparcV9.burm.cpp

