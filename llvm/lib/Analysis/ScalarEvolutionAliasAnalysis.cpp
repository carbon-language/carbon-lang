//===- ScalarEvolutionAliasAnalysis.cpp - SCEV-based Alias Analysis -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScalarEvolutionAliasAnalysis pass, which implements a
// simple alias analysis implemented in terms of ScalarEvolution queries.
//
// This differs from traditional loop dependence analysis in that it tests
// for dependencies within a single iteration of a loop, rather than
// dependencies between different iterations.
//
// ScalarEvolution has a more complete understanding of pointer arithmetic
// than BasicAliasAnalysis' collection of ad-hoc analyses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  /// ScalarEvolutionAliasAnalysis - This is a simple alias analysis
  /// implementation that uses ScalarEvolution to answer queries.
  class ScalarEvolutionAliasAnalysis : public FunctionPass,
                                       public AliasAnalysis {
    ScalarEvolution *SE;

  public:
    static char ID; // Class identification, replacement for typeinfo
    ScalarEvolutionAliasAnalysis() : FunctionPass(&ID), SE(0) {}

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const PassInfo *PI) {
      if (PI->isPassID(&AliasAnalysis::ID))
        return (AliasAnalysis*)this;
      return this;
    }

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnFunction(Function &F);
    virtual AliasResult alias(const Value *V1, unsigned V1Size,
                              const Value *V2, unsigned V2Size);

    Value *GetBaseValue(const SCEV *S);
  };
}  // End of anonymous namespace

// Register this pass...
char ScalarEvolutionAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(ScalarEvolutionAliasAnalysis, AliasAnalysis, "scev-aa",
                   "ScalarEvolution-based Alias Analysis", false, true, false);

FunctionPass *llvm::createScalarEvolutionAliasAnalysisPass() {
  return new ScalarEvolutionAliasAnalysis();
}

void
ScalarEvolutionAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<ScalarEvolution>();
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

bool
ScalarEvolutionAliasAnalysis::runOnFunction(Function &F) {
  InitializeAliasAnalysis(this);
  SE = &getAnalysis<ScalarEvolution>();
  return false;
}

/// GetBaseValue - Given an expression, try to find a
/// base value. Return null is none was found.
Value *
ScalarEvolutionAliasAnalysis::GetBaseValue(const SCEV *S) {
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // In an addrec, assume that the base will be in the start, rather
    // than the step.
    return GetBaseValue(AR->getStart());
  } else if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(S)) {
    // If there's a pointer operand, it'll be sorted at the end of the list.
    const SCEV *Last = A->getOperand(A->getNumOperands()-1);
    if (Last->getType()->isPointerTy())
      return GetBaseValue(Last);
  } else if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // This is a leaf node.
    return U->getValue();
  }
  // No Identified object found.
  return 0;
}

AliasAnalysis::AliasResult
ScalarEvolutionAliasAnalysis::alias(const Value *A, unsigned ASize,
                                    const Value *B, unsigned BSize) {
  // If either of the memory references is empty, it doesn't matter what the
  // pointer values are. This allows the code below to ignore this special
  // case.
  if (ASize == 0 || BSize == 0)
    return NoAlias;

  // This is ScalarEvolutionAliasAnalysis. Get the SCEVs!
  const SCEV *AS = SE->getSCEV(const_cast<Value *>(A));
  const SCEV *BS = SE->getSCEV(const_cast<Value *>(B));

  // If they evaluate to the same expression, it's a MustAlias.
  if (AS == BS) return MustAlias;

  // If something is known about the difference between the two addresses,
  // see if it's enough to prove a NoAlias.
  if (SE->getEffectiveSCEVType(AS->getType()) ==
      SE->getEffectiveSCEVType(BS->getType())) {
    unsigned BitWidth = SE->getTypeSizeInBits(AS->getType());
    APInt ASizeInt(BitWidth, ASize);
    APInt BSizeInt(BitWidth, BSize);

    // Compute the difference between the two pointers.
    const SCEV *BA = SE->getMinusSCEV(BS, AS);

    // Test whether the difference is known to be great enough that memory of
    // the given sizes don't overlap. This assumes that ASizeInt and BSizeInt
    // are non-zero, which is special-cased above.
    if (ASizeInt.ule(SE->getUnsignedRange(BA).getUnsignedMin()) &&
        (-BSizeInt).uge(SE->getUnsignedRange(BA).getUnsignedMax()))
      return NoAlias;

    // Folding the subtraction while preserving range information can be tricky
    // (because of INT_MIN, etc.); if the prior test failed, swap AS and BS
    // and try again to see if things fold better that way.

    // Compute the difference between the two pointers.
    const SCEV *AB = SE->getMinusSCEV(AS, BS);

    // Test whether the difference is known to be great enough that memory of
    // the given sizes don't overlap. This assumes that ASizeInt and BSizeInt
    // are non-zero, which is special-cased above.
    if (BSizeInt.ule(SE->getUnsignedRange(AB).getUnsignedMin()) &&
        (-ASizeInt).uge(SE->getUnsignedRange(AB).getUnsignedMax()))
      return NoAlias;
  }

  // If ScalarEvolution can find an underlying object, form a new query.
  // The correctness of this depends on ScalarEvolution not recognizing
  // inttoptr and ptrtoint operators.
  Value *AO = GetBaseValue(AS);
  Value *BO = GetBaseValue(BS);
  if ((AO && AO != A) || (BO && BO != B))
    if (alias(AO ? AO : A, AO ? UnknownSize : ASize,
              BO ? BO : B, BO ? UnknownSize : BSize) == NoAlias)
      return NoAlias;

  // Forward the query to the next analysis.
  return AliasAnalysis::alias(A, ASize, B, BSize);
}
