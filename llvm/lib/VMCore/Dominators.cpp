//===- Dominators.cpp - Dominator Calculation -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators.  Postdominators are available in libanalysis, but are not
// included in libvmcore, because it's not needed.  Forward dominators are
// needed to support the Verifier pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

// Always verify dominfo if expensive checking is enabled.
#ifdef XDEBUG
static bool VerifyDomInfo = true;
#else
static bool VerifyDomInfo = false;
#endif
static cl::opt<bool,true>
VerifyDomInfoX("verify-dom-info", cl::location(VerifyDomInfo),
               cl::desc("Verify dominator info (time consuming)"));

//===----------------------------------------------------------------------===//
//  DominatorTree Implementation
//===----------------------------------------------------------------------===//
//
// Provide public access to DominatorTree information.  Implementation details
// can be found in DominatorInternals.h.
//
//===----------------------------------------------------------------------===//

TEMPLATE_INSTANTIATION(class llvm::DomTreeNodeBase<BasicBlock>);
TEMPLATE_INSTANTIATION(class llvm::DominatorTreeBase<BasicBlock>);

char DominatorTree::ID = 0;
INITIALIZE_PASS(DominatorTree, "domtree",
                "Dominator Tree Construction", true, true)

bool DominatorTree::runOnFunction(Function &F) {
  DT->recalculate(F);
  return false;
}

void DominatorTree::verifyAnalysis() const {
  if (!VerifyDomInfo) return;

  Function &F = *getRoot()->getParent();

  DominatorTree OtherDT;
  OtherDT.getBase().recalculate(F);
  if (compare(OtherDT)) {
    errs() << "DominatorTree is not up to date!\nComputed:\n";
    print(errs());
    errs() << "\nActual:\n";
    OtherDT.print(errs());
    abort();
  }
}

void DominatorTree::print(raw_ostream &OS, const Module *) const {
  DT->print(OS);
}

// dominates - Return true if Def dominates a use in User. This performs
// the special checks necessary if Def and User are in the same basic block.
// Note that Def doesn't dominate a use in Def itself!
bool DominatorTree::dominates(const Instruction *Def,
                              const Instruction *User) const {
  const BasicBlock *UseBB = User->getParent();
  const BasicBlock *DefBB = Def->getParent();

  // Any unreachable use is dominated, even if Def == User.
  if (!isReachableFromEntry(UseBB))
    return true;

  // Unreachable definitions don't dominate anything.
  if (!isReachableFromEntry(DefBB))
    return false;

  // An instruction doesn't dominate a use in itself.
  if (Def == User)
    return false;

  // The value defined by an invoke dominates an instruction only if
  // it dominates every instruction in UseBB.
  // A PHI is dominated only if the instruction dominates every possible use
  // in the UseBB.
  if (isa<InvokeInst>(Def) || isa<PHINode>(User))
    return dominates(Def, UseBB);

  if (DefBB != UseBB)
    return dominates(DefBB, UseBB);

  // Loop through the basic block until we find Def or User.
  BasicBlock::const_iterator I = DefBB->begin();
  for (; &*I != Def && &*I != User; ++I)
    /*empty*/;

  return &*I == Def;
}

// true if Def would dominate a use in any instruction in UseBB.
// note that dominates(Def, Def->getParent()) is false.
bool DominatorTree::dominates(const Instruction *Def,
                              const BasicBlock *UseBB) const {
  const BasicBlock *DefBB = Def->getParent();

  // Any unreachable use is dominated, even if DefBB == UseBB.
  if (!isReachableFromEntry(UseBB))
    return true;

  // Unreachable definitions don't dominate anything.
  if (!isReachableFromEntry(DefBB))
    return false;

  if (DefBB == UseBB)
    return false;

  const InvokeInst *II = dyn_cast<InvokeInst>(Def);
  if (!II)
    return dominates(DefBB, UseBB);

  // Invoke results are only usable in the normal destination, not in the
  // exceptional destination.
  BasicBlock *NormalDest = II->getNormalDest();
  if (!dominates(NormalDest, UseBB))
    return false;

  // Simple case: if the normal destination has a single predecessor, the
  // fact that it dominates the use block implies that we also do.
  if (NormalDest->getSinglePredecessor())
    return true;

  // The normal edge from the invoke is critical. Conceptually, what we would
  // like to do is split it and check if the new block dominates the use.
  // With X being the new block, the graph would look like:
  //
  //        DefBB
  //          /\      .  .
  //         /  \     .  .
  //        /    \    .  .
  //       /      \   |  |
  //      A        X  B  C
  //      |         \ | /
  //      .          \|/
  //      .      NormalDest
  //      .
  //
  // Given the definition of dominance, NormalDest is dominated by X iff X
  // dominates all of NormalDest's predecessors (X, B, C in the example). X
  // trivially dominates itself, so we only have to find if it dominates the
  // other predecessors. Since the only way out of X is via NormalDest, X can
  // only properly dominate a node if NormalDest dominates that node too.
  for (pred_iterator PI = pred_begin(NormalDest),
         E = pred_end(NormalDest); PI != E; ++PI) {
    const BasicBlock *BB = *PI;
    if (BB == DefBB)
      continue;

    if (!DT->isReachableFromEntry(BB))
      continue;

    if (!dominates(NormalDest, BB))
      return false;
  }
  return true;
}
