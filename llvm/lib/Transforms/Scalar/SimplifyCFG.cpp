//===- SimplifyCFG.cpp - CFG Simplification Pass --------------------------===//
//
// This file implements dead code elimination and basic block merging.
//
// Specifically, this:
//   * removes basic blocks with no predecessors
//   * merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor
//   * Eliminates a basic block that only contains an unconditional branch
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Module.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/CFG.h"
#include "llvm/Pass.h"
#include "Support/StatisticReporter.h"
#include <set>

static Statistic<> NumSimpl("cfgsimplify\t- Number of blocks simplified");

namespace {
  struct CFGSimplifyPass : public FunctionPass {
    const char *getPassName() const { return "Simplify CFG"; }
    
    virtual bool runOnFunction(Function *F);
  };
}

Pass *createCFGSimplificationPass() {
  return new CFGSimplifyPass();
}

static bool MarkAliveBlocks(BasicBlock *BB, std::set<BasicBlock*> &Reachable) {
  if (Reachable.count(BB)) return false;
  Reachable.insert(BB);

  bool Changed = ConstantFoldTerminator(BB);
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    MarkAliveBlocks(*SI, Reachable);

  return Changed;
}


// It is possible that we may require multiple passes over the code to fully
// simplify the CFG.
//
bool CFGSimplifyPass::runOnFunction(Function *F) {
  std::set<BasicBlock*> Reachable;
  bool Changed = MarkAliveBlocks(F->front(), Reachable);

  // If there are unreachable blocks in the CFG...
  if (Reachable.size() != F->size()) {
    assert(Reachable.size() < F->size());
    NumSimpl += F->size()-Reachable.size();

    // Loop over all of the basic blocks that are not reachable, dropping all of
    // their internal references...
    for (Function::iterator I = F->begin()+1, E = F->end(); I != E; ++I)
      if (!Reachable.count(*I)) {
        BasicBlock *BB = *I;
        for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI!=SE; ++SI)
          if (Reachable.count(*SI))
            (*SI)->removePredecessor(BB);
        BB->dropAllReferences();
      }
    
    for (Function::iterator I = F->begin()+1; I != F->end();)
      if (!Reachable.count(*I))
        delete F->getBasicBlocks().remove(I);
      else
        ++I;

    Changed = true;
  }

  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks (except the first one) and remove them
    // if they are unneeded...
    //
    for (Function::iterator BBIt = F->begin()+1; BBIt != F->end(); ) {
      if (SimplifyCFG(BBIt)) {
        LocalChange = true;
        ++NumSimpl;
      } else {
        ++BBIt;
      }
    }
    Changed |= LocalChange;
  }

  return Changed;
}
