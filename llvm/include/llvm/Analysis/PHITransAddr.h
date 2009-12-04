//===- PHITransAddr.h - PHI Translation for Addresses -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PHITransAddr class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PHITRANSADDR_H
#define LLVM_ANALYSIS_PHITRANSADDR_H

#include "llvm/Instruction.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class DominatorTree;
  class TargetData;
  
/// PHITransAddr - An address value which tracks and handles phi translation.
/// As we walk "up" the CFG through predecessors, we need to ensure that the
/// address we're tracking is kept up to date.  For example, if we're analyzing
/// an address of "&A[i]" and walk through the definition of 'i' which is a PHI
/// node, we *must* phi translate i to get "&A[j]" or else we will analyze an
/// incorrect pointer in the predecessor block.
///
/// This is designed to be a relatively small object that lives on the stack and
/// is copyable.
///
class PHITransAddr {
  /// Addr - The actual address we're analyzing.
  Value *Addr;
  
  /// InstInputs - The inputs for our symbolic address.
  SmallVector<Instruction*, 4> InstInputs;
public:
  PHITransAddr(Value *addr) : Addr(addr) {
    // If the address is an instruction, the whole thing is considered an input.
    if (Instruction *I = dyn_cast<Instruction>(Addr))
      InstInputs.push_back(I);
  }
  
  /// NeedsPHITranslationFromBlock - Return true if moving from the specified
  /// BasicBlock to its predecessors requires PHI translation.
  bool NeedsPHITranslationFromBlock(BasicBlock *BB) const {
    // We do need translation if one of our input instructions is defined in
    // this block.
    for (unsigned i = 0, e = InstInputs.size(); i != e; ++i)
      if (InstInputs[i]->getParent() == BB)
        return true;
    return false;
  }
  
  /// IsPHITranslatable - If this needs PHI translation, return true if we have
  /// some hope of doing it.  This should be used as a filter to avoid calling
  /// GetPHITranslatedValue in hopeless situations.
  bool IsPHITranslatable() const;
  
  /// GetPHITranslatedValue - Given a computation that satisfied the
  /// isPHITranslatable predicate, see if we can translate the computation into
  /// the specified predecessor block.  If so, return that value, otherwise
  /// return null.
  Value *GetPHITranslatedValue(Value *InVal, BasicBlock *CurBB,
                               BasicBlock *Pred, const TargetData *TD) const;
  
  /// GetAvailablePHITranslatePointer - Return the value computed by
  /// PHITranslatePointer if it dominates PredBB, otherwise return null.
  Value *GetAvailablePHITranslatedValue(Value *V,
                                        BasicBlock *CurBB, BasicBlock *PredBB,
                                        const TargetData *TD,
                                        const DominatorTree &DT) const;
  
  /// InsertPHITranslatedPointer - Insert a computation of the PHI translated
  /// version of 'V' for the edge PredBB->CurBB into the end of the PredBB
  /// block.  All newly created instructions are added to the NewInsts list.
  /// This returns null on failure.
  ///
  Value *InsertPHITranslatedPointer(Value *InVal, BasicBlock *CurBB,
                                    BasicBlock *PredBB, const TargetData *TD,
                                    const DominatorTree &DT,
                                 SmallVectorImpl<Instruction*> &NewInsts) const;
    
};

} // end namespace llvm

#endif
