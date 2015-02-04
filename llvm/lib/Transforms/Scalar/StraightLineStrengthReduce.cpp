//===-- StraightLineStrengthReduce.cpp - ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements straight-line strength reduction (SLSR). Unlike loop
// strength reduction, this algorithm is designed to reduce arithmetic
// redundancy in straight-line code instead of loops. It has proven to be
// effective in simplifying arithmetic statements derived from an unrolled loop.
// It can also simplify the logic of SeparateConstOffsetFromGEP.
//
// There are many optimizations we can perform in the domain of SLSR. This file
// for now contains only an initial step. Specifically, we look for strength
// reduction candidate in the form of
//
// (B + i) * S
//
// where B and S are integer constants or variables, and i is a constant
// integer. If we found two such candidates
//
// S1: X = (B + i) * S S2: Y = (B + i') * S
//
// and S1 dominates S2, we call S1 a basis of S2, and can replace S2 with
//
// Y = X + (i' - i) * S
//
// where (i' - i) * S is folded to the extent possible. When S2 has multiple
// bases, we pick the one that is closest to S2, or S2's "immediate" basis.
//
// TODO:
//
// - Handle candidates in the form of B + i * S
//
// - Handle candidates in the form of pointer arithmetics. e.g., B[i * S]
//
// - Floating point arithmetics when fast math is enabled.
//
// - SLSR may decrease ILP at the architecture level. Targets that are very
//   sensitive to ILP may want to disable it. Having SLSR to consider ILP is
//   left as future work.
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace PatternMatch;

namespace {

class StraightLineStrengthReduce : public FunctionPass {
 public:
  // SLSR candidate. Such a candidate must be in the form of
  //   (Base + Index) * Stride
  struct Candidate : public ilist_node<Candidate> {
    Candidate(Value *B = nullptr, ConstantInt *Idx = nullptr,
              Value *S = nullptr, Instruction *I = nullptr)
        : Base(B), Index(Idx), Stride(S), Ins(I), Basis(nullptr) {}
    Value *Base;
    ConstantInt *Index;
    Value *Stride;
    // The instruction this candidate corresponds to. It helps us to rewrite a
    // candidate with respect to its immediate basis. Note that one instruction
    // can corresponds to multiple candidates depending on how you associate the
    // expression. For instance,
    //
    // (a + 1) * (b + 2)
    //
    // can be treated as
    //
    // <Base: a, Index: 1, Stride: b + 2>
    //
    // or
    //
    // <Base: b, Index: 2, Stride: a + 1>
    Instruction *Ins;
    // Points to the immediate basis of this candidate, or nullptr if we cannot
    // find any basis for this candidate.
    Candidate *Basis;
  };

  static char ID;

  StraightLineStrengthReduce() : FunctionPass(ID), DT(nullptr) {
    initializeStraightLineStrengthReducePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    // We do not modify the shape of the CFG.
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override;

 private:
  // Returns true if Basis is a basis for C, i.e., Basis dominates C and they
  // share the same base and stride.
  bool isBasisFor(const Candidate &Basis, const Candidate &C);
  // Checks whether I is in a candidate form. If so, adds all the matching forms
  // to Candidates, and tries to find the immediate basis for each of them.
  void allocateCandidateAndFindBasis(Instruction *I);
  // Given that I is in the form of "(B + Idx) * S", adds this form to
  // Candidates, and finds its immediate basis.
  void allocateCandidateAndFindBasis(Value *B, ConstantInt *Idx, Value *S,
                                     Instruction *I);
  // Rewrites candidate C with respect to Basis.
  void rewriteCandidateWithBasis(const Candidate &C, const Candidate &Basis);

  DominatorTree *DT;
  ilist<Candidate> Candidates;
  // Temporarily holds all instructions that are unlinked (but not deleted) by
  // rewriteCandidateWithBasis. These instructions will be actually removed
  // after all rewriting finishes.
  DenseSet<Instruction *> UnlinkedInstructions;
};
}  // anonymous namespace

char StraightLineStrengthReduce::ID = 0;
INITIALIZE_PASS_BEGIN(StraightLineStrengthReduce, "slsr",
                      "Straight line strength reduction", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(StraightLineStrengthReduce, "slsr",
                    "Straight line strength reduction", false, false)

FunctionPass *llvm::createStraightLineStrengthReducePass() {
  return new StraightLineStrengthReduce();
}

bool StraightLineStrengthReduce::isBasisFor(const Candidate &Basis,
                                            const Candidate &C) {
  return (Basis.Ins != C.Ins && // skip the same instruction
          // Basis must dominate C in order to rewrite C with respect to Basis.
          DT->dominates(Basis.Ins->getParent(), C.Ins->getParent()) &&
          // They share the same base and stride.
          Basis.Base == C.Base &&
          Basis.Stride == C.Stride);
}

// TODO: We currently implement an algorithm whose time complexity is linear to
// the number of existing candidates. However, a better algorithm exists. We
// could depth-first search the dominator tree, and maintain a hash table that
// contains all candidates that dominate the node being traversed.  This hash
// table is indexed by the base and the stride of a candidate.  Therefore,
// finding the immediate basis of a candidate boils down to one hash-table look
// up.
void StraightLineStrengthReduce::allocateCandidateAndFindBasis(Value *B,
                                                               ConstantInt *Idx,
                                                               Value *S,
                                                               Instruction *I) {
  Candidate C(B, Idx, S, I);
  // Try to compute the immediate basis of C.
  unsigned NumIterations = 0;
  // Limit the scan radius to avoid running forever.
  static const unsigned MaxNumIterations = 50;
  for (auto Basis = Candidates.rbegin();
       Basis != Candidates.rend() && NumIterations < MaxNumIterations;
       ++Basis, ++NumIterations) {
    if (isBasisFor(*Basis, C)) {
      C.Basis = &(*Basis);
      break;
    }
  }
  // Regardless of whether we find a basis for C, we need to push C to the
  // candidate list.
  Candidates.push_back(C);
}

void StraightLineStrengthReduce::allocateCandidateAndFindBasis(Instruction *I) {
  Value *B = nullptr;
  ConstantInt *Idx = nullptr;
  // "(Base + Index) * Stride" must be a Mul instruction at the first hand.
  if (I->getOpcode() == Instruction::Mul) {
    if (IntegerType *ITy = dyn_cast<IntegerType>(I->getType())) {
      Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
      for (unsigned Swapped = 0; Swapped < 2; ++Swapped) {
        // Only handle the canonical operand ordering.
        if (match(LHS, m_Add(m_Value(B), m_ConstantInt(Idx)))) {
          // If LHS is in the form of "Base + Index", then I is in the form of
          // "(Base + Index) * RHS".
          allocateCandidateAndFindBasis(B, Idx, RHS, I);
        } else {
          // Otherwise, at least try the form (LHS + 0) * RHS.
          allocateCandidateAndFindBasis(LHS, ConstantInt::get(ITy, 0), RHS, I);
        }
        // Swap LHS and RHS so that we also cover the cases where LHS is the
        // stride.
        if (LHS == RHS)
          break;
        std::swap(LHS, RHS);
      }
    }
  }
}

void StraightLineStrengthReduce::rewriteCandidateWithBasis(
    const Candidate &C, const Candidate &Basis) {
  // An instruction can correspond to multiple candidates. Therefore, instead of
  // simply deleting an instruction when we rewrite it, we mark its parent as
  // nullptr (i.e. unlink it) so that we can skip the candidates whose
  // instruction is already rewritten.
  if (!C.Ins->getParent())
    return;
  assert(C.Base == Basis.Base && C.Stride == Basis.Stride);
  // Basis = (B + i) * S
  // C     = (B + i') * S
  //   ==>
  // C     = Basis + (i' - i) * S
  IRBuilder<> Builder(C.Ins);
  ConstantInt *IndexOffset = ConstantInt::get(
      C.Ins->getContext(), C.Index->getValue() - Basis.Index->getValue());
  Value *Reduced;
  // TODO: preserve nsw/nuw in some cases.
  if (IndexOffset->isOne()) {
    // If (i' - i) is 1, fold C into Basis + S.
    Reduced = Builder.CreateAdd(Basis.Ins, C.Stride);
  } else if (IndexOffset->isMinusOne()) {
    // If (i' - i) is -1, fold C into Basis - S.
    Reduced = Builder.CreateSub(Basis.Ins, C.Stride);
  } else {
    Value *Bump = Builder.CreateMul(C.Stride, IndexOffset);
    Reduced = Builder.CreateAdd(Basis.Ins, Bump);
  }
  Reduced->takeName(C.Ins);
  C.Ins->replaceAllUsesWith(Reduced);
  C.Ins->dropAllReferences();
  // Unlink C.Ins so that we can skip other candidates also corresponding to
  // C.Ins. The actual deletion is postponed to the end of runOnFunction.
  C.Ins->removeFromParent();
  UnlinkedInstructions.insert(C.Ins);
}

bool StraightLineStrengthReduce::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  // Traverse the dominator tree in the depth-first order. This order makes sure
  // all bases of a candidate are in Candidates when we process it.
  for (auto node = GraphTraits<DominatorTree *>::nodes_begin(DT);
       node != GraphTraits<DominatorTree *>::nodes_end(DT); ++node) {
    BasicBlock *B = node->getBlock();
    for (auto I = B->begin(); I != B->end(); ++I) {
      allocateCandidateAndFindBasis(I);
    }
  }

  // Rewrite candidates in the reverse depth-first order. This order makes sure
  // a candidate being rewritten is not a basis for any other candidate.
  while (!Candidates.empty()) {
    const Candidate &C = Candidates.back();
    if (C.Basis != nullptr) {
      rewriteCandidateWithBasis(C, *C.Basis);
    }
    Candidates.pop_back();
  }

  // Delete all unlink instructions.
  for (auto I : UnlinkedInstructions) {
    delete I;
  }
  bool Ret = !UnlinkedInstructions.empty();
  UnlinkedInstructions.clear();
  return Ret;
}
