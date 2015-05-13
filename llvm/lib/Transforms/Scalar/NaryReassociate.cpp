//===- NaryReassociate.cpp - Reassociate n-ary expressions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates n-ary add expressions and eliminates the redundancy
// exposed by the reassociation.
//
// A motivating example:
//
//   void foo(int a, int b) {
//     bar(a + b);
//     bar((a + 2) + b);
//   }
//
// An ideal compiler should reassociate (a + 2) + b to (a + b) + 2 and simplify
// the above code to
//
//   int t = a + b;
//   bar(t);
//   bar(t + 2);
//
// However, the Reassociate pass is unable to do that because it processes each
// instruction individually and believes (a + 2) + b is the best form according
// to its rank system.
//
// To address this limitation, NaryReassociate reassociates an expression in a
// form that reuses existing instructions. As a result, NaryReassociate can
// reassociate (a + 2) + b in the example to (a + b) + 2 because it detects that
// (a + b) is computed before.
//
// NaryReassociate works as follows. For every instruction in the form of (a +
// b) + c, it checks whether a + c or b + c is already computed by a dominating
// instruction. If so, it then reassociates (a + b) + c into (a + c) + b or (b +
// c) + a and removes the redundancy accordingly. To efficiently look up whether
// an expression is computed before, we store each instruction seen and its SCEV
// into an SCEV-to-instruction map.
//
// Although the algorithm pattern-matches only ternary additions, it
// automatically handles many >3-ary expressions by walking through the function
// in the depth-first order. For example, given
//
//   (a + c) + d
//   ((a + b) + c) + d
//
// NaryReassociate first rewrites (a + b) + c to (a + c) + b, and then rewrites
// ((a + c) + b) + d into ((a + c) + d) + b.
//
// Finally, the above dominator-based algorithm may need to be run multiple
// iterations before emitting optimal code. One source of this need is that we
// only split an operand when it is used only once. The above algorithm can
// eliminate an instruction and decrease the usage count of its operands. As a
// result, an instruction that previously had multiple uses may become a
// single-use instruction and thus eligible for split consideration. For
// example,
//
//   ac = a + c
//   ab = a + b
//   abc = ab + c
//   ab2 = ab + b
//   ab2c = ab2 + c
//
// In the first iteration, we cannot reassociate abc to ac+b because ab is used
// twice. However, we can reassociate ab2c to abc+b in the first iteration. As a
// result, ab2 becomes dead and ab will be used only once in the second
// iteration.
//
// Limitations and TODO items:
//
// 1) We only considers n-ary adds for now. This should be extended and
// generalized.
//
// 2) Besides arithmetic operations, similar reassociation can be applied to
// GEPs. For example, if
//   X = &arr[a]
// dominates
//   Y = &arr[a + b]
// we may rewrite Y into X + b.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "nary-reassociate"

namespace {
class NaryReassociate : public FunctionPass {
public:
  static char ID;

  NaryReassociate(): FunctionPass(ID) {
    initializeNaryReassociatePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<ScalarEvolution>();
    AU.addPreserved<TargetLibraryInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  // Runs only one iteration of the dominator-based algorithm. See the header
  // comments for why we need multiple iterations.
  bool doOneIteration(Function &F);
  // Reasssociates I to a better form.
  Instruction *tryReassociateAdd(Instruction *I);
  // A helper function for tryReassociateAdd. LHS and RHS are explicitly passed.
  Instruction *tryReassociateAdd(Value *LHS, Value *RHS, Instruction *I);
  // Rewrites I to LHS + RHS if LHS is computed already.
  Instruction *tryReassociatedAdd(const SCEV *LHS, Value *RHS, Instruction *I);

  DominatorTree *DT;
  ScalarEvolution *SE;
  TargetLibraryInfo *TLI;
  // A lookup table quickly telling which instructions compute the given SCEV.
  // Note that there can be multiple instructions at different locations
  // computing to the same SCEV, so we map a SCEV to an instruction list.  For
  // example,
  //
  //   if (p1)
  //     foo(a + b);
  //   if (p2)
  //     bar(a + b);
  DenseMap<const SCEV *, SmallVector<Instruction *, 2>> SeenExprs;
};
} // anonymous namespace

char NaryReassociate::ID = 0;
INITIALIZE_PASS_BEGIN(NaryReassociate, "nary-reassociate", "Nary reassociation",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(NaryReassociate, "nary-reassociate", "Nary reassociation",
                    false, false)

FunctionPass *llvm::createNaryReassociatePass() {
  return new NaryReassociate();
}

bool NaryReassociate::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = &getAnalysis<ScalarEvolution>();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  bool Changed = false, ChangedInThisIteration;
  do {
    ChangedInThisIteration = doOneIteration(F);
    Changed |= ChangedInThisIteration;
  } while (ChangedInThisIteration);
  return Changed;
}

bool NaryReassociate::doOneIteration(Function &F) {
  bool Changed = false;
  SeenExprs.clear();
  // Traverse the dominator tree in the depth-first order. This order makes sure
  // all bases of a candidate are in Candidates when we process it.
  for (auto Node = GraphTraits<DominatorTree *>::nodes_begin(DT);
       Node != GraphTraits<DominatorTree *>::nodes_end(DT); ++Node) {
    BasicBlock *BB = Node->getBlock();
    for (auto I = BB->begin(); I != BB->end(); ++I) {
      // Skip vector types which are not SCEVable.
      if (I->getOpcode() == Instruction::Add && !I->getType()->isVectorTy()) {
        if (Instruction *NewI = tryReassociateAdd(I)) {
          Changed = true;
          SE->forgetValue(I);
          I->replaceAllUsesWith(NewI);
          RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
          I = NewI;
        }
        // We should add the rewritten instruction because tryReassociateAdd may
        // have invalidated the original one.
        SeenExprs[SE->getSCEV(I)].push_back(I);
      }
    }
  }
  return Changed;
}

Instruction *NaryReassociate::tryReassociateAdd(Instruction *I) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  if (auto *NewI = tryReassociateAdd(LHS, RHS, I))
    return NewI;
  if (auto *NewI = tryReassociateAdd(RHS, LHS, I))
    return NewI;
  return nullptr;
}

Instruction *NaryReassociate::tryReassociateAdd(Value *LHS, Value *RHS,
                                                Instruction *I) {
  Value *A = nullptr, *B = nullptr;
  // To be conservative, we reassociate I only when it is the only user of A+B.
  if (LHS->hasOneUse() && match(LHS, m_Add(m_Value(A), m_Value(B)))) {
    // I = (A + B) + RHS
    //   = (A + RHS) + B or (B + RHS) + A
    const SCEV *AExpr = SE->getSCEV(A), *BExpr = SE->getSCEV(B);
    const SCEV *RHSExpr = SE->getSCEV(RHS);
    if (BExpr != RHSExpr) {
      if (auto *NewI = tryReassociatedAdd(SE->getAddExpr(AExpr, RHSExpr), B, I))
        return NewI;
    }
    if (AExpr != RHSExpr) {
      if (auto *NewI = tryReassociatedAdd(SE->getAddExpr(BExpr, RHSExpr), A, I))
        return NewI;
    }
  }
  return nullptr;
}

Instruction *NaryReassociate::tryReassociatedAdd(const SCEV *LHSExpr,
                                                 Value *RHS, Instruction *I) {
  auto Pos = SeenExprs.find(LHSExpr);
  // Bail out if LHSExpr is not previously seen.
  if (Pos == SeenExprs.end())
    return nullptr;

  auto &LHSCandidates = Pos->second;
  // Look for the closest dominator LHS of I that computes LHSExpr, and replace
  // I with LHS + RHS.
  //
  // Because we traverse the dominator tree in the pre-order, a
  // candidate that doesn't dominate the current instruction won't dominate any
  // future instruction either. Therefore, we pop it out of the stack. This
  // optimization makes the algorithm O(n).
  while (!LHSCandidates.empty()) {
    Instruction *LHS = LHSCandidates.back();
    if (DT->dominates(LHS, I)) {
      Instruction *NewI = BinaryOperator::CreateAdd(LHS, RHS, "", I);
      NewI->takeName(I);
      return NewI;
    }
    LHSCandidates.pop_back();
  }
  return nullptr;
}
