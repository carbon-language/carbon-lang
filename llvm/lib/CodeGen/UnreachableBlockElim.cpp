//===-- UnreachableBlockElim.cpp - Remove unreachable blocks for codegen --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is an extremely simple version of the SimplifyCFG pass.  Its sole
// job is to delete LLVM basic blocks that are not reachable from the entry
// node.  To do this, it performs a simple depth first traversal of the CFG,
// then deletes any unvisited nodes.
//
// Note that this pass is really a hack.  In particular, the instruction
// selectors for various targets should just not generate code for unreachable
// blocks.  Until LLVM has a more systematic way of defining instruction
// selectors, however, we cannot really expect them to handle additional
// complexity.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DepthFirstIterator.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN UnreachableBlockElim : public FunctionPass {
    virtual bool runOnFunction(Function &F);
  };
  RegisterOpt<UnreachableBlockElim>
  X("unreachableblockelim", "Remove unreachable blocks from the CFG");
}

FunctionPass *llvm::createUnreachableBlockEliminationPass() {
  return new UnreachableBlockElim();
}

bool UnreachableBlockElim::runOnFunction(Function &F) {
  std::set<BasicBlock*> Reachable;

  // Mark all reachable blocks.
  for (df_ext_iterator<Function*> I = df_ext_begin(&F, Reachable),
         E = df_ext_end(&F, Reachable); I != E; ++I)
    /* Mark all reachable blocks */;

  // Loop over all dead blocks, remembering them and deleting all instructions
  // in them.
  std::vector<BasicBlock*> DeadBlocks;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (!Reachable.count(I)) {
      BasicBlock *BB = I;
      DeadBlocks.push_back(BB);
      while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
        PN->replaceAllUsesWith(Constant::getNullValue(PN->getType()));
        BB->getInstList().pop_front();
      }
      for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
        (*SI)->removePredecessor(BB);
      BB->dropAllReferences();
    }

  if (DeadBlocks.empty()) return false;

  // Actually remove the blocks now.
  for (unsigned i = 0, e = DeadBlocks.size(); i != e; ++i)
    F.getBasicBlockList().erase(DeadBlocks[i]);

  return true;
}
