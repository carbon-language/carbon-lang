//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
// This file implements dead inst elimination and dead code elimination.
//
// Dead Inst Elimination performs a single pass over the function removing
// instructions that are obviously dead.  Dead Code Elimination is similar, but
// it rechecks instructions that were used by removed instructions to see if
// they are newly dead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Pass.h"
#include "llvm/InstrTypes.h"
#include "llvm/Function.h"
#include "llvm/Support/InstIterator.h"
#include <set>

static inline bool isInstDead(Instruction *I) {
  return I->use_empty() && !I->hasSideEffects() && !isa<TerminatorInst>(I);
}

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool dceInstruction(BasicBlock::InstListType &BBIL,
                    BasicBlock::iterator &BBI) {
  // Look for un"used" definitions...
  if (isInstDead(*BBI)) {
    delete BBIL.remove(BBI);   // Bye bye
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// DeadInstElimination pass implementation
//

namespace {
  struct DeadInstElimination : public BasicBlockPass {
    const char *getPassName() const { return "Dead Instruction Elimination"; }
    
    virtual bool runOnBasicBlock(BasicBlock *BB) {
      BasicBlock::InstListType &Vals = BB->getInstList();
      bool Changed = false;
      for (BasicBlock::iterator DI = Vals.begin(); DI != Vals.end(); )
        if (dceInstruction(Vals, DI))
          Changed = true;
        else
          ++DI;
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  };
}

Pass *createDeadInstEliminationPass() {
  return new DeadInstElimination();
}



//===----------------------------------------------------------------------===//
// DeadCodeElimination pass implementation
//

namespace {
  struct DCE : public FunctionPass {
    const char *getPassName() const { return "Dead Code Elimination"; }

    virtual bool runOnFunction(Function *F);

     virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
 };
}

bool DCE::runOnFunction(Function *F) {
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
    
    if (isInstDead(I)) {       // If the instruction is dead...
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
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    BasicBlock::InstListType &BBIL = (*I)->getInstList();
    for (BasicBlock::iterator BI = BBIL.begin(); BI != BBIL.end(); )
      if (DeadInsts.count(*BI)) {            // Is this instruction dead?
        delete BBIL.remove(BI);              // Yup, remove and delete inst
      } else {                               // This instruction is not dead
        ++BI;                                // Continue on to the next one...
      }
  }

  return true;
}

Pass *createDeadCodeEliminationPass() {
  return new DCE();
}
