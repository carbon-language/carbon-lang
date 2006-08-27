//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead inst elimination and dead code elimination.
//
// Dead Inst Elimination performs a single pass over the function removing
// instructions that are obviously dead.  Dead Code Elimination is similar, but
// it rechecks instructions that were used by removed instructions to see if
// they are newly dead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> DIEEliminated("die", "Number of insts removed");
  Statistic<> DCEEliminated("dce", "Number of insts removed");

  //===--------------------------------------------------------------------===//
  // DeadInstElimination pass implementation
  //

  struct DeadInstElimination : public BasicBlockPass {
    virtual bool runOnBasicBlock(BasicBlock &BB) {
      bool Changed = false;
      for (BasicBlock::iterator DI = BB.begin(); DI != BB.end(); )
        if (dceInstruction(DI)) {
          Changed = true;
          ++DIEEliminated;
        } else
          ++DI;
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };

  RegisterPass<DeadInstElimination> X("die", "Dead Instruction Elimination");
}

FunctionPass *llvm::createDeadInstEliminationPass() {
  return new DeadInstElimination();
}


//===----------------------------------------------------------------------===//
// DeadCodeElimination pass implementation
//

namespace {
  struct DCE : public FunctionPass {
    virtual bool runOnFunction(Function &F);

     virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
 };

  RegisterPass<DCE> Y("dce", "Dead Code Elimination");
}

bool DCE::runOnFunction(Function &F) {
  // Start out with all of the instructions in the worklist...
  std::vector<Instruction*> WorkList;
  for (inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i)
    WorkList.push_back(&*i);

  // Loop over the worklist finding instructions that are dead.  If they are
  // dead make them drop all of their uses, making other instructions
  // potentially dead, and work until the worklist is empty.
  //
  bool MadeChange = false;
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();
    WorkList.pop_back();

    if (isInstructionTriviallyDead(I)) {       // If the instruction is dead.
      // Loop over all of the values that the instruction uses, if there are
      // instructions being used, add them to the worklist, because they might
      // go dead after this one is removed.
      //
      for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
        if (Instruction *Used = dyn_cast<Instruction>(*OI))
          WorkList.push_back(Used);

      // Remove the instruction.
      I->eraseFromParent();

      // Remove the instruction from the worklist if it still exists in it.
      for (std::vector<Instruction*>::iterator WI = WorkList.begin(),
             E = WorkList.end(); WI != E; ++WI)
        if (*WI == I) {
          WorkList.erase(WI);
          --E;
          --WI;
        }

      MadeChange = true;
      ++DCEEliminated;
    }
  }
  return MadeChange;
}

FunctionPass *llvm::createDeadCodeEliminationPass() {
  return new DCE();
}

