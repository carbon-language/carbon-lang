#===-- lib/CodeGen/Makefile.am -----------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

SUBDIRS = SelectionDAG

libexec_PROGRAMS = LLVMCodeGen.o

LLVMCodeGen_o_SOURCES = \
  AsmPrinter.cpp \
  BranchFolding.cpp \
  IntrinsicLowering.cpp \
  LiveIntervalAnalysis.cpp \
  LiveInterval.cpp \
  LiveVariables.cpp \
  MachineBasicBlock.cpp \
  MachineCodeEmitter.cpp \
  MachineFunction.cpp \
  MachineInstr.cpp \
  Passes.cpp \
  PHIElimination.cpp \
  PrologEpilogInserter.cpp \
  RegAllocIterativeScan.cpp \
  RegAllocLinearScan.cpp \
  RegAllocLocal.cpp \
  RegAllocSimple.cpp \
  TwoAddressInstructionPass.cpp \
  UnreachableBlockElim.cpp \
  VirtRegMap.cpp

LIBS=
