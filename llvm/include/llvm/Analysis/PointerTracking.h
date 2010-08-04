//===- PointerTracking.h - Pointer Bounds Tracking --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements tracking of pointer bounds.
// It knows that the libc functions "calloc" and "realloc" allocate memory, thus
// you should avoid using this pass if they mean something else for your
// language.
//
// All methods assume that the pointer is not NULL, if it is then the returned
// allocation size is wrong, and the result from checkLimits is wrong too.
// It also assumes that pointers are valid, and that it is not analyzing a
// use-after-free scenario.
// Due to these limitations the "size" returned by these methods should be
// considered as either 0 or the returned size.
//
// Another analysis pass should be used to find use-after-free/NULL dereference
// bugs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POINTERTRACKING_H
#define LLVM_ANALYSIS_POINTERTRACKING_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/PredIteratorCache.h"

namespace llvm {
  class DominatorTree;
  class ScalarEvolution;
  class SCEV;
  class Loop;
  class LoopInfo;
  class TargetData;

  // Result from solver, assuming pointer is not NULL,
  // and it is not a use-after-free situation.
  enum SolverResult {
    AlwaysFalse,// always false with above constraints
    AlwaysTrue,// always true with above constraints
    Unknown // it can sometimes be true, sometimes false, or it is undecided
  };

  class PointerTracking : public FunctionPass {
  public:
    typedef ICmpInst::Predicate Predicate;
    static char ID;
    PointerTracking();

    virtual bool doInitialization(Module &M);

    // If this pointer directly points to an allocation, return
    // the number of elements of type Ty allocated.
    // Otherwise return CouldNotCompute.
    // Since allocations can fail by returning NULL, the real element count
    // for every allocation is either 0 or the value returned by this function.
    const SCEV *getAllocationElementCount(Value *P) const;

    // Same as getAllocationSize() but returns size in bytes.
    // We consider one byte as 8 bits.
    const SCEV *getAllocationSizeInBytes(Value *V) const;

    // Given a Pointer, determine a base pointer of known size, and an offset
    // therefrom.
    // When unable to determine, sets Base to NULL, and Limit/Offset to
    // CouldNotCompute.
    // BaseSize, and Offset are in bytes: Pointer == Base + Offset
    void getPointerOffset(Value *Pointer, Value *&Base, const SCEV *& BaseSize,
                          const SCEV *&Offset) const;

    // Compares the 2 scalar evolution expressions according to predicate,
    // and if it can prove that the result is always true or always false
    // return AlwaysTrue/AlwaysFalse. Otherwise it returns Unknown.
    enum SolverResult compareSCEV(const SCEV *A, Predicate Pred, const SCEV *B,
                                  const Loop *L);

    // Determines whether the condition LHS <Pred> RHS is sufficient
    // for the condition A <Pred> B to hold.
    // Currently only ULT/ULE is supported.
    // This errs on the side of returning false.
    bool conditionSufficient(const SCEV *LHS, Predicate Pred1, const SCEV *RHS,
                             const SCEV *A, Predicate Pred2, const SCEV *B,
                             const Loop *L);

    // Determines whether Offset is known to be always in [0, Limit) bounds.
    // This errs on the side of returning Unknown.
    enum SolverResult checkLimits(const SCEV *Offset, const SCEV *Limit,
                                  BasicBlock *BB);

    virtual bool runOnFunction(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    void print(raw_ostream &OS, const Module* = 0) const;
    Value *computeAllocationCountValue(Value *P, const Type *&Ty) const;
  private:
    Function *FF;
    TargetData *TD;
    ScalarEvolution *SE;
    LoopInfo *LI;
    DominatorTree *DT;

    Function *callocFunc;
    Function *reallocFunc;
    PredIteratorCache predCache;

    SmallPtrSet<const SCEV*, 1> analyzing;

    enum SolverResult isLoopGuardedBy(const Loop *L, Predicate Pred,
                                      const SCEV *A, const SCEV *B) const;
    static bool isMonotonic(const SCEV *S);
    bool scevPositive(const SCEV *A, const Loop *L, bool strict=true) const;
    bool conditionSufficient(Value *Cond, bool negated,
                             const SCEV *A, Predicate Pred, const SCEV *B);
    Value *getConditionToReach(BasicBlock *A,
                               DomTreeNodeBase<BasicBlock> *B,
                               bool &negated);
    Value *getConditionToReach(BasicBlock *A,
                               BasicBlock *B,
                               bool &negated);
    const SCEV *computeAllocationCount(Value *P, const Type *&Ty) const;
    const SCEV *computeAllocationCountForType(Value *P, const Type *Ty) const;
  };
}
#endif

