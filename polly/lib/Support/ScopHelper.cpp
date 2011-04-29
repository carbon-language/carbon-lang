//===- ScopHelper.cpp - Some Helper Functions for Scop.  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with Scop and LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ScopHelper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "polly-scop-helper"
#include "llvm/Support/Debug.h"

using namespace llvm;


namespace {
// Checks if a SCEV is invariant in a region. This is if all Values are
// referenced in this SCEV are defined outside the region.
class InvariantChecker: SCEVVisitor<InvariantChecker, bool> {
  Region &R;

public:
  bool visitConstant(const SCEVConstant *S) {
    return true;
  }

  bool visitUnknown(const SCEVUnknown* S) {
    Value *V = S->getValue();

    // An Instruction defined outside the region is invariant.
    if (Instruction *I = dyn_cast<Instruction>(V))
      return !R.contains(I);

    // A constant is invariant.
    return true;
  }

  bool visitNAryExpr(const SCEVNAryExpr *S) {
    for (SCEVNAryExpr::op_iterator OI = S->op_begin(), OE = S->op_end();
         OI != OE; ++OI)
      if (!visit(*OI))
        return false;

    return true;
  }

  bool visitMulExpr(const SCEVMulExpr* S) {
    return visitNAryExpr(S);
  }

  bool visitCastExpr(const SCEVCastExpr *S) {
    return visit(S->getOperand());
  }

  bool visitTruncateExpr(const SCEVTruncateExpr *S) {
    return visit(S->getOperand());
  }

  bool visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
    return visit(S->getOperand());
  }

  bool visitSignExtendExpr(const SCEVSignExtendExpr *S) {
    return visit(S->getOperand());
  }

  bool visitAddExpr(const SCEVAddExpr *S) {
    return visitNAryExpr(S);
  }

  bool visitAddRecExpr(const SCEVAddRecExpr *S) {
    // Check if the addrec is contained in the region.
    if (R.contains(S->getLoop()))
      return false;

    return visitNAryExpr(S);
  }

  bool visitUDivExpr(const SCEVUDivExpr *S) {
    return visit(S->getLHS()) && visit(S->getRHS());
  }

  bool visitSMaxExpr(const SCEVSMaxExpr *S) {
    return visitNAryExpr(S);
  }

  bool visitUMaxExpr(const SCEVUMaxExpr *S) {
    return visitNAryExpr(S);
  }

  bool visitCouldNotCompute(const SCEVCouldNotCompute *S) {
    llvm_unreachable("SCEV cannot be checked");
  }

  InvariantChecker(Region &RefRegion)
    : R(RefRegion) {}

  static bool isInvariantInRegion(const SCEV *S, Region &R) {
    InvariantChecker Checker(R);
    return Checker.visit(S);
  }
};
}

// Helper function for Scop
// TODO: Add assertion to not allow parameter to be null
//===----------------------------------------------------------------------===//
// Temporary Hack for extended region tree.
// Cast the region to loop if there is a loop have the same header and exit.
Loop *polly::castToLoop(const Region &R, LoopInfo &LI) {
  BasicBlock *entry = R.getEntry();

  if (!LI.isLoopHeader(entry))
    return 0;

  Loop *L = LI.getLoopFor(entry);

  BasicBlock *exit = L->getExitBlock();

  // Is the loop with multiple exits?
  if (!exit) return 0;

  if (exit != R.getExit()) {
    // SubRegion/ParentRegion with the same entry.
    assert((R.getNode(R.getEntry())->isSubRegion()
            || R.getParent()->getEntry() == entry)
           && "Expect the loop is the smaller or bigger region");
    return 0;
  }

  return L;
}

Value *polly::getPointerOperand(Instruction &Inst) {
  if (LoadInst *load = dyn_cast<LoadInst>(&Inst))
    return load->getPointerOperand();
  else if (StoreInst *store = dyn_cast<StoreInst>(&Inst))
    return store->getPointerOperand();
  else if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&Inst))
    return gep->getPointerOperand();

  return 0;
}

//===----------------------------------------------------------------------===//
// Helper functions

bool polly::isInvariant(const SCEV *S, Region &R) {
  return InvariantChecker::isInvariantInRegion(S, R);
}

// Helper function to check parameter
bool polly::isParameter(const SCEV *Var, Region &RefRegion,
                        LoopInfo &LI, ScalarEvolution &SE) {
  assert(Var && "Var can not be null!");

  if (!isInvariant(Var, RefRegion))
    return false;

  if (isa<SCEVAddRecExpr>(Var))
    return true;

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(Var)) {
    if (isa<PHINode>(U->getValue()))
      return false;

    if(isa<UndefValue>(U->getValue()))
      return false;

    return true;
  }

  if (const SCEVCastExpr *Cast = dyn_cast<SCEVCastExpr>(Var))
    return isParameter(Cast->getOperand(), RefRegion, LI, SE);

  return false;
}

bool polly::isIndVar(const SCEV *Var, Region &RefRegion,
                     LoopInfo &LI, ScalarEvolution &SE) {
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Var);

  // AddRecExprs are no induction variables.
  if (!AddRec) return false;

  Loop *L = const_cast<Loop*>(AddRec->getLoop());

  // Is the addrec an induction variable of a loop contained in the current
  // region.
  if (!RefRegion.contains(L))
    return false;

  DEBUG(dbgs() << "Find AddRec: " << *AddRec
        << " at region: " << RefRegion.getNameStr() << " as indvar\n");
  return true;
}

bool polly::isIndVar(const Instruction *I, const LoopInfo *LI) {
  Loop *L = LI->getLoopFor(I->getParent());

  return L && I == L->getCanonicalInductionVariable();
}

bool polly::hasInvokeEdge(const PHINode *PN) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
    if (InvokeInst *II = dyn_cast<InvokeInst>(PN->getIncomingValue(i)))
      if (II->getParent() == PN->getIncomingBlock(i))
        return true;

  return false;
}

// Helper function for LLVM-IR about Scop
BasicBlock *polly::createSingleEntryEdge(Region *R, Pass *P) {
  BasicBlock *BB = R->getEntry();

  BasicBlock::iterator SplitIt = BB->begin();

  while (isa<PHINode>(SplitIt))
    ++SplitIt;

  BasicBlock *newBB = SplitBlock(BB, SplitIt, P);

  for (BasicBlock::iterator PI = BB->begin(); isa<PHINode>(PI); ++PI) {
    PHINode *PN = cast<PHINode>(PI);
    PHINode *NPN =
      PHINode::Create(PN->getType(), 2, PN->getName()+".ph", newBB->begin());

    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
      if (R->contains(*PI)) {
        Value *V = PN->removeIncomingValue(*PI, false);
        NPN->addIncoming(V, *PI);
      }
    }
    PN->replaceAllUsesWith(NPN);
    NPN->addIncoming(PN,BB);
  }

  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (R->contains(*PI))
      (*PI)->getTerminator()->replaceUsesOfWith(BB, newBB);

  return newBB;
}

BasicBlock *polly::createSingleExitEdge(Region *R, Pass *P) {
  BasicBlock *BB = R->getExit();

  SmallVector<BasicBlock*, 4> Preds;
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (R->contains(*PI))
      Preds.push_back(*PI);

  return SplitBlockPredecessors(BB, Preds.data(), Preds.size(), ".region", P);
}

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, Pass *P) {
  // Find first non-alloca instruction. Every basic block has a non-alloc
  // instruction, as every well formed basic block has a terminator.
  BasicBlock::iterator I = EntryBlock->begin();
  while (isa<AllocaInst>(I)) ++I;

  // SplitBlock updates DT, DF and LI.
  BasicBlock *NewEntry = SplitBlock(EntryBlock, I, P);
  if (RegionInfo *RI = P->getAnalysisIfAvailable<RegionInfo>())
    RI->splitBlock(NewEntry, EntryBlock);
}
