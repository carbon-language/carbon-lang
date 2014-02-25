//===- SimplifyCFGPass.cpp - CFG Simplification Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging, along
// with a collection of other peephole control flow optimizations.  For example:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//   * Changes invoke instructions to nounwind functions to be calls.
//   * Change things like "if (x) if (y)" into "if (x&y)".
//   * etc..
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplifycfg"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

STATISTIC(NumSimpl, "Number of blocks simplified");

namespace {
struct CFGSimplifyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  CFGSimplifyPass() : FunctionPass(ID) {
    initializeCFGSimplifyPassPass(*PassRegistry::getPassRegistry());
  }
  virtual bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<TargetTransformInfo>();
  }
};
}

char CFGSimplifyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                      false)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_END(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                    false)

// Public interface to the CFGSimplification pass
FunctionPass *llvm::createCFGSimplificationPass() {
  return new CFGSimplifyPass();
}

/// mergeEmptyReturnBlocks - If we have more than one empty (other than phi
/// node) return blocks, merge them together to promote recursive block merging.
static bool mergeEmptyReturnBlocks(Function &F) {
  bool Changed = false;

  BasicBlock *RetBlock = 0;

  // Scan all the blocks in the function, looking for empty return blocks.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; ) {
    BasicBlock &BB = *BBI++;

    // Only look at return blocks.
    ReturnInst *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (Ret == 0) continue;

    // Only look at the block if it is empty or the only other thing in it is a
    // single PHI node that is the operand to the return.
    if (Ret != &BB.front()) {
      // Check for something else in the block.
      BasicBlock::iterator I = Ret;
      --I;
      // Skip over debug info.
      while (isa<DbgInfoIntrinsic>(I) && I != BB.begin())
        --I;
      if (!isa<DbgInfoIntrinsic>(I) &&
          (!isa<PHINode>(I) || I != BB.begin() ||
           Ret->getNumOperands() == 0 ||
           Ret->getOperand(0) != I))
        continue;
    }

    // If this is the first returning block, remember it and keep going.
    if (RetBlock == 0) {
      RetBlock = &BB;
      continue;
    }

    // Otherwise, we found a duplicate return block.  Merge the two.
    Changed = true;

    // Case when there is no input to the return or when the returned values
    // agree is trivial.  Note that they can't agree if there are phis in the
    // blocks.
    if (Ret->getNumOperands() == 0 ||
        Ret->getOperand(0) ==
          cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0)) {
      BB.replaceAllUsesWith(RetBlock);
      BB.eraseFromParent();
      continue;
    }

    // If the canonical return block has no PHI node, create one now.
    PHINode *RetBlockPHI = dyn_cast<PHINode>(RetBlock->begin());
    if (RetBlockPHI == 0) {
      Value *InVal = cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0);
      pred_iterator PB = pred_begin(RetBlock), PE = pred_end(RetBlock);
      RetBlockPHI = PHINode::Create(Ret->getOperand(0)->getType(),
                                    std::distance(PB, PE), "merge",
                                    &RetBlock->front());

      for (pred_iterator PI = PB; PI != PE; ++PI)
        RetBlockPHI->addIncoming(InVal, *PI);
      RetBlock->getTerminator()->setOperand(0, RetBlockPHI);
    }

    // Turn BB into a block that just unconditionally branches to the return
    // block.  This handles the case when the two return blocks have a common
    // predecessor but that return different things.
    RetBlockPHI->addIncoming(Ret->getOperand(0), &BB);
    BB.getTerminator()->eraseFromParent();
    BranchInst::Create(RetBlock, &BB);
  }

  return Changed;
}

/// iterativelySimplifyCFG - Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool iterativelySimplifyCFG(Function &F, const TargetTransformInfo &TTI,
                                   const DataLayout *DL) {
  bool Changed = false;
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks and remove them if they are unneeded...
    //
    for (Function::iterator BBIt = F.begin(); BBIt != F.end(); ) {
      if (SimplifyCFG(BBIt++, TTI, DL)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

// It is possible that we may require multiple passes over the code to fully
// simplify the CFG.
//
bool CFGSimplifyPass::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  const TargetTransformInfo &TTI = getAnalysis<TargetTransformInfo>();
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  const DataLayout *DL = DLP ? &DLP->getDataLayout() : 0;
  bool EverChanged = removeUnreachableBlocks(F);
  EverChanged |= mergeEmptyReturnBlocks(F);
  EverChanged |= iterativelySimplifyCFG(F, TTI, DL);

  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // iterativelySimplifyCFG can (rarely) make some loops dead.  If this happens,
  // removeUnreachableBlocks is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid reruning iterativelySimplifyCFG if the second pass of
  // removeUnreachableBlocks doesn't do anything.
  if (!removeUnreachableBlocks(F))
    return true;

  do {
    EverChanged = iterativelySimplifyCFG(F, TTI, DL);
    EverChanged |= removeUnreachableBlocks(F);
  } while (EverChanged);

  return true;
}
