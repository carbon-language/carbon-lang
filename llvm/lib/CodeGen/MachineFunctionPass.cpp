//===-- MachineFunctionPass.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionPass members.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LiveValues.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

bool MachineFunctionPass::runOnFunction(Function &F) {
  // Do not codegen any 'available_externally' functions at all, they have
  // definitions outside the translation unit.
  if (F.hasAvailableExternallyLinkage())
    return false;

  MachineFunction &MF = getAnalysis<MachineFunctionAnalysis>().getMF();
  return runOnMachineFunction(MF);
}

void MachineFunctionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineFunctionAnalysis>();
  AU.addPreserved<MachineFunctionAnalysis>();

  // MachineFunctionPass preserves all LLVM IR passes, but there's no
  // high-level way to express this. Instead, just list a bunch of
  // passes explicitly.
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<ScalarEvolution>();
  AU.addPreserved<IVUsers>();
  AU.addPreserved<MemoryDependenceAnalysis>();
  AU.addPreserved<LiveValues>();
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<DominanceFrontier>();
  AU.addPreserved<LoopInfo>();

  FunctionPass::getAnalysisUsage(AU);
}
