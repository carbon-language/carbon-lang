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
#include "Support/Statistic.h"
#include <set>

namespace llvm {

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
  
  RegisterOpt<DeadInstElimination> X("die", "Dead Instruction Elimination");
}

Pass *createDeadInstEliminationPass() {
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

  RegisterOpt<DCE> Y("dce", "Dead Code Elimination");
}

bool DCE::runOnFunction(Function &F) {
  // Start out with all of the instructions in the worklist...
  std::vector<Instruction*> WorkList(inst_begin(F), inst_end(F));
  std::set<Instruction*> DeadInsts;
  
  // Loop over the worklist finding instructions that are dead.  If they are
  // dead make them drop all of their uses, making other instructions
  // potentially dead, and work until the worklist is empty.
  //
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();
    WorkList.pop_back();
    
    if (isInstructionTriviallyDead(I)) {       // If the instruction is dead...
      // Loop over all of the values that the instruction uses, if there are
      // instructions being used, add them to the worklist, because they might
      // go dead after this one is removed.
      //
      for (User::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI)
        if (Instruction *Used = dyn_cast<Instruction>(*UI))
          WorkList.push_back(Used);

      // Tell the instruction to let go of all of the values it uses...
      I->dropAllReferences();

      // Keep track of this instruction, because we are going to delete it later
      DeadInsts.insert(I);
    }
  }

  // If we found no dead instructions, we haven't changed the function...
  if (DeadInsts.empty()) return false;

  // Otherwise, loop over the program, removing and deleting the instructions...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (BasicBlock::iterator BI = I->begin(); BI != I->end(); )
      if (DeadInsts.count(BI)) {             // Is this instruction dead?
        BI = I->getInstList().erase(BI);     // Yup, remove and delete inst
        ++DCEEliminated;
      } else {                               // This instruction is not dead
        ++BI;                                // Continue on to the next one...
      }

  return true;
}

Pass *createDeadCodeEliminationPass() {
  return new DCE();
}

} // End llvm namespace
