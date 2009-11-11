//===- LiveValues.cpp - Liveness information for LLVM IR Values. ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the implementation for the LLVM IR Value liveness
// analysis pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LiveValues.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
using namespace llvm;

namespace llvm {
  FunctionPass *createLiveValuesPass() { return new LiveValues(); }
}

char LiveValues::ID = 0;
static RegisterPass<LiveValues>
X("live-values", "Value Liveness Analysis", false, true);

LiveValues::LiveValues() : FunctionPass(&ID) {}

void LiveValues::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.setPreservesAll();
}

bool LiveValues::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  // This pass' values are computed lazily, so there's nothing to do here.

  return false;
}

void LiveValues::releaseMemory() {
  Memos.clear();
}

/// isUsedInBlock - Test if the given value is used in the given block.
///
bool LiveValues::isUsedInBlock(const Value *V, const BasicBlock *BB) {
  Memo &M = getMemo(V);
  return M.Used.count(BB);
}

/// isLiveThroughBlock - Test if the given value is known to be
/// live-through the given block, meaning that the block is properly
/// dominated by the value's definition, and there exists a block
/// reachable from it that contains a use. This uses a conservative
/// approximation that errs on the side of returning false.
///
bool LiveValues::isLiveThroughBlock(const Value *V,
                                    const BasicBlock *BB) {
  Memo &M = getMemo(V);
  return M.LiveThrough.count(BB);
}

/// isKilledInBlock - Test if the given value is known to be killed in
/// the given block, meaning that the block contains a use of the value,
/// and no blocks reachable from the block contain a use. This uses a
/// conservative approximation that errs on the side of returning false.
///
bool LiveValues::isKilledInBlock(const Value *V, const BasicBlock *BB) {
  Memo &M = getMemo(V);
  return M.Killed.count(BB);
}

/// getMemo - Retrieve an existing Memo for the given value if one
/// is available, otherwise compute a new one.
///
LiveValues::Memo &LiveValues::getMemo(const Value *V) {
  DenseMap<const Value *, Memo>::iterator I = Memos.find(V);
  if (I != Memos.end())
    return I->second;
  return compute(V);
}

/// getImmediateDominator - A handy utility for the specific DominatorTree
/// query that we need here.
///
static const BasicBlock *getImmediateDominator(const BasicBlock *BB,
                                               const DominatorTree *DT) {
  DomTreeNode *Node = DT->getNode(const_cast<BasicBlock *>(BB))->getIDom();
  return Node ? Node->getBlock() : 0;
}

/// compute - Compute a new Memo for the given value.
///
LiveValues::Memo &LiveValues::compute(const Value *V) {
  Memo &M = Memos[V];

  // Determine the block containing the definition.
  const BasicBlock *DefBB;
  // Instructions define values with meaningful live ranges.
  if (const Instruction *I = dyn_cast<Instruction>(V))
    DefBB = I->getParent();
  // Arguments can be analyzed as values defined in the entry block.
  else if (const Argument *A = dyn_cast<Argument>(V))
    DefBB = &A->getParent()->getEntryBlock();
  // Constants and other things aren't meaningful here, so just
  // return having computed an empty Memo so that we don't come
  // here again. The assumption here is that client code won't
  // be asking about such values very often.
  else
    return M;

  // Determine if the value is defined inside a loop. This is used
  // to track whether the value is ever used outside the loop, so
  // it'll be set to null if the value is either not defined in a
  // loop or used outside the loop in which it is defined.
  const Loop *L = LI->getLoopFor(DefBB);

  // Track whether the value is used anywhere outside of the block
  // in which it is defined.
  bool LiveOutOfDefBB = false;

  // Examine each use of the value.
  for (Value::use_const_iterator I = V->use_begin(), E = V->use_end();
       I != E; ++I) {
    const User *U = *I;
    const BasicBlock *UseBB = cast<Instruction>(U)->getParent();

    // Note the block in which this use occurs.
    M.Used.insert(UseBB);

    // If the use block doesn't have successors, the value can be
    // considered killed.
    if (succ_begin(UseBB) == succ_end(UseBB))
      M.Killed.insert(UseBB);

    // Observe whether the value is used outside of the loop in which
    // it is defined. Switch to an enclosing loop if necessary.
    for (; L; L = L->getParentLoop())
      if (L->contains(UseBB))
        break;

    // Search for live-through blocks.
    const BasicBlock *BB;
    if (const PHINode *PHI = dyn_cast<PHINode>(U)) {
      // For PHI nodes, start the search at the incoming block paired with the
      // incoming value, which must be dominated by the definition.
      unsigned Num = PHI->getIncomingValueNumForOperand(I.getOperandNo());
      BB = PHI->getIncomingBlock(Num);

      // A PHI-node use means the value is live-out of it's defining block
      // even if that block also contains the only use.
      LiveOutOfDefBB = true;
    } else {
      // Otherwise just start the search at the use.
      BB = UseBB;

      // Note if the use is outside the defining block.
      LiveOutOfDefBB |= UseBB != DefBB;
    }

    // Climb the immediate dominator tree from the use to the definition
    // and mark all intermediate blocks as live-through.
    for (; BB != DefBB; BB = getImmediateDominator(BB, DT)) {
      if (BB != UseBB && !M.LiveThrough.insert(BB))
        break;
    }
  }

  // If the value is defined inside a loop and is not live outside
  // the loop, then each exit block of the loop in which the value
  // is used is a kill block.
  if (L) {
    SmallVector<BasicBlock *, 4> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      const BasicBlock *ExitingBlock = ExitingBlocks[i];
      if (M.Used.count(ExitingBlock))
        M.Killed.insert(ExitingBlock);
    }
  }

  // If the value was never used outside the the block in which it was
  // defined, it's killed in that block.
  if (!LiveOutOfDefBB)
    M.Killed.insert(DefBB);

  return M;
}
