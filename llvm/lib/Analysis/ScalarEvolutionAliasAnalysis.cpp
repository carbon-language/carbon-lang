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
// ScalarEvolution has a more complete understanding of pointer arithmetic
// than BasicAliasAnalysis' collection of ad-hoc analyses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  /// ScalarEvolutionAliasAnalysis - This is a simple alias analysis
  /// implementation that uses ScalarEvolution to answer queries.
  class VISIBILITY_HIDDEN ScalarEvolutionAliasAnalysis : public FunctionPass,
                                                         public AliasAnalysis {
    ScalarEvolution *SE;

  public:
    static char ID; // Class identification, replacement for typeinfo
    ScalarEvolutionAliasAnalysis() : FunctionPass(&ID), SE(0) {}

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnFunction(Function &F);
    virtual AliasResult alias(const Value *V1, unsigned V1Size,
                              const Value *V2, unsigned V2Size);

    Value *GetUnderlyingIdentifiedObject(const SCEV *S);
  };
}  // End of anonymous namespace

// Register this pass...
char ScalarEvolutionAliasAnalysis::ID = 0;
static RegisterPass<ScalarEvolutionAliasAnalysis>
X("scev-aa", "ScalarEvolution-based Alias Analysis", false, true);

// Declare that we implement the AliasAnalysis interface
static RegisterAnalysisGroup<AliasAnalysis> Y(X);

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

Value *
ScalarEvolutionAliasAnalysis::GetUnderlyingIdentifiedObject(const SCEV *S) {
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    return GetUnderlyingIdentifiedObject(AR->getStart());
  } else if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(S)) {
    // If there's a pointer operand, it'll be sorted at the end of the list.
    const SCEV *Last = A->getOperand(A->getNumOperands()-1);
    if (isa<PointerType>(Last->getType()))
      return GetUnderlyingIdentifiedObject(Last);
  } else if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // Determine if we've found an Identified object.
    Value *V = U->getValue();
    if (isIdentifiedObject(V))
      return V;
  }
  // No Identified object found.
  return 0;
}

AliasAnalysis::AliasResult
ScalarEvolutionAliasAnalysis::alias(const Value *A, unsigned ASize,
                                    const Value *B, unsigned BSize) {
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
    APInt AI(BitWidth, ASize);
    const SCEV *BA = SE->getMinusSCEV(BS, AS);
    if (AI.ule(SE->getUnsignedRange(BA).getUnsignedMin())) {
      APInt BI(BitWidth, BSize);
      const SCEV *AB = SE->getMinusSCEV(AS, BS);
      if (BI.ule(SE->getUnsignedRange(AB).getUnsignedMin()))
        return NoAlias;
    }
  }

  // If ScalarEvolution can find an underlying object, form a new query.
  // The correctness of this depends on ScalarEvolution not recognizing
  // inttoptr and ptrtoint operators.
  Value *AO = GetUnderlyingIdentifiedObject(AS);
  Value *BO = GetUnderlyingIdentifiedObject(BS);
  if ((AO && AO != A) || (BO && BO != B))
    if (alias(AO ? AO : A, AO ? ~0u : ASize,
              BO ? BO : B, BO ? ~0u : BSize) == NoAlias)
      return NoAlias;

  // Forward the query to the next analysis.
  return AliasAnalysis::alias(A, ASize, B, BSize);
}
