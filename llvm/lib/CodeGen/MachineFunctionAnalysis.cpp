//===-- MachineFunctionAnalysis.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionAnalysis members.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
using namespace llvm;

// Register this pass with PassInfo directly to avoid having to define
// a default constructor.
static PassInfo
X("Machine Function Analysis", "machine-function-analysis",
  intptr_t(&MachineFunctionAnalysis::ID), 0,
  /*CFGOnly=*/false, /*is_analysis=*/true);

char MachineFunctionAnalysis::ID = 0;

MachineFunctionAnalysis::MachineFunctionAnalysis(TargetMachine &tm,
                                                 CodeGenOpt::Level OL) :
  FunctionPass(&ID), TM(tm), OptLevel(OL), MF(0) {
}

MachineFunctionAnalysis::~MachineFunctionAnalysis() {
  releaseMemory();
  assert(!MF && "MachineFunctionAnalysis left initialized!");
}

bool MachineFunctionAnalysis::runOnFunction(Function &F) {
  assert(!MF && "MachineFunctionAnalysis already initialized!");
  MF = new MachineFunction(&F, TM);
  return false;
}

void MachineFunctionAnalysis::releaseMemory() {
  delete MF;
  MF = 0;
}

void MachineFunctionAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}
