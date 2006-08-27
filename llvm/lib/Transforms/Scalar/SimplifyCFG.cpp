//===- SimplifyCFG.cpp - CFG Simplification Pass --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CFG.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumSimpl("cfgsimplify", "Number of blocks simplified");

  struct CFGSimplifyPass : public FunctionPass {
    virtual bool runOnFunction(Function &F);
  };
  RegisterPass<CFGSimplifyPass> X("simplifycfg", "Simplify the CFG");
}

// Public interface to the CFGSimplification pass
FunctionPass *llvm::createCFGSimplificationPass() {
  return new CFGSimplifyPass();
}

static bool MarkAliveBlocks(BasicBlock *BB, std::set<BasicBlock*> &Reachable) {
  if (Reachable.count(BB)) return false;
  Reachable.insert(BB);

  // Do a quick scan of the basic block, turning any obviously unreachable
  // instructions into LLVM unreachable insts.  The instruction combining pass
  // canonnicalizes unreachable insts into stores to null or undef.
  for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ++BBI)
    if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (isa<ConstantPointerNull>(SI->getOperand(1)) ||
          isa<UndefValue>(SI->getOperand(1))) {
        // Loop over all of the successors, removing BB's entry from any PHI
        // nodes.
        for (succ_iterator I = succ_begin(BB), SE = succ_end(BB); I != SE; ++I)
          (*I)->removePredecessor(BB);

        new UnreachableInst(SI);

        // All instructions after this are dead.
        for (; BBI != E; ) {
          if (!BBI->use_empty())
            BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
          BB->getInstList().erase(BBI++);
        }
        break;
      }


  bool Changed = ConstantFoldTerminator(BB);
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    MarkAliveBlocks(*SI, Reachable);

  return Changed;
}


// It is possible that we may require multiple passes over the code to fully
// simplify the CFG.
//
bool CFGSimplifyPass::runOnFunction(Function &F) {
  std::set<BasicBlock*> Reachable;
  bool Changed = MarkAliveBlocks(F.begin(), Reachable);

  // If there are unreachable blocks in the CFG...
  if (Reachable.size() != F.size()) {
    assert(Reachable.size() < F.size());
    NumSimpl += F.size()-Reachable.size();

    // Loop over all of the basic blocks that are not reachable, dropping all of
    // their internal references...
    for (Function::iterator BB = ++F.begin(), E = F.end(); BB != E; ++BB)
      if (!Reachable.count(BB)) {
        for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI!=SE; ++SI)
          if (Reachable.count(*SI))
            (*SI)->removePredecessor(BB);
        BB->dropAllReferences();
      }

    for (Function::iterator I = ++F.begin(); I != F.end();)
      if (!Reachable.count(I))
        I = F.getBasicBlockList().erase(I);
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
    for (Function::iterator BBIt = ++F.begin(); BBIt != F.end(); ) {
      if (SimplifyCFG(BBIt++)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }

  return Changed;
}
