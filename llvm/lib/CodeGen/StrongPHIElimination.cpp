//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

namespace {
class StrongPHIElimination : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  StrongPHIElimination() : MachineFunctionPass(ID) {
    initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
  }

private:
  bool runOnMachineFunction(MachineFunction &Fn) {
    llvm_unreachable("Strong phi elimination is not implemented");
  }
    
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<SlotIndexes>();
    AU.addPreserved<SlotIndexes>();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char StrongPHIElimination::ID = 0;
INITIALIZE_PASS_BEGIN(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)

char &llvm::StrongPHIEliminationID = StrongPHIElimination::ID;

