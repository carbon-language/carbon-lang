//===- PHITransAddr.cpp - PHI Translation for Addresses -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PHITransAddr class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/Dominators.h"
using namespace llvm;

/// IsPHITranslatable - If this needs PHI translation, return true if we have
/// some hope of doing it.  This should be used as a filter to avoid calling
/// GetPHITranslatedValue in hopeless situations.
bool PHITransAddr::IsPHITranslatable() const {
  return true; // not a good filter.
}

/// GetPHITranslatedValue - Given a computation that satisfied the
/// isPHITranslatable predicate, see if we can translate the computation into
/// the specified predecessor block.  If so, return that value, otherwise
/// return null.
Value *PHITransAddr::GetPHITranslatedValue(Value *InVal, BasicBlock *CurBB,
                                           BasicBlock *Pred,
                                           const TargetData *TD) const {
  // Not a great implementation.
  return 0;
}

/// GetAvailablePHITranslatePointer - Return the value computed by
/// PHITranslatePointer if it dominates PredBB, otherwise return null.
Value *PHITransAddr::
GetAvailablePHITranslatedValue(Value *V,
                               BasicBlock *CurBB, BasicBlock *PredBB,
                               const TargetData *TD,
                               const DominatorTree &DT) const {
  // See if PHI translation succeeds.
  V = GetPHITranslatedValue(V, CurBB, PredBB, TD);
  if (V == 0) return 0;
  
  // Make sure the value is live in the predecessor.
  if (Instruction *Inst = dyn_cast_or_null<Instruction>(V))
    if (!DT.dominates(Inst->getParent(), PredBB))
      return 0;
  return V;
}

/// InsertPHITranslatedPointer - Insert a computation of the PHI translated
/// version of 'V' for the edge PredBB->CurBB into the end of the PredBB
/// block.  All newly created instructions are added to the NewInsts list.
/// This returns null on failure.
///
Value *PHITransAddr::
InsertPHITranslatedPointer(Value *InVal, BasicBlock *CurBB,
                           BasicBlock *PredBB, const TargetData *TD,
                           const DominatorTree &DT,
                           SmallVectorImpl<Instruction*> &NewInsts) const {
  // See if we have a version of this value already available and dominating
  // PredBB.  If so, there is no need to insert a new copy.
  if (Value *Res = GetAvailablePHITranslatedValue(InVal, CurBB, PredBB, TD, DT))
    return Res;

  // Not a great implementation.
  return 0;
}
