//===- PtrUseVisitor.cpp - InstVisitors over a pointers uses --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// Implementation of the pointer use visitors.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PtrUseVisitor.h"

using namespace llvm;

void detail::PtrUseVisitorBase::enqueueUsers(Instruction &I) {
  for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
       UI != UE; ++UI) {
    if (VisitedUses.insert(&UI.getUse())) {
      UseToVisit NewU = {
        UseToVisit::UseAndIsOffsetKnownPair(&UI.getUse(), IsOffsetKnown),
        Offset
      };
      Worklist.push_back(llvm_move(NewU));
    }
  }
}

bool detail::PtrUseVisitorBase::adjustOffsetForGEP(GetElementPtrInst &GEPI) {
  if (!IsOffsetKnown)
    return false;

  for (gep_type_iterator GTI = gep_type_begin(GEPI), GTE = gep_type_end(GEPI);
       GTI != GTE; ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
    if (!OpC)
      return false;
    if (OpC->isZero())
      continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = DL.getStructLayout(STy);
      Offset += APInt(Offset.getBitWidth(),
                         SL->getElementOffset(ElementIdx));
      continue;
    }

    // For array or vector indices, scale the index by the size of the type.
    APInt Index = OpC->getValue().sextOrTrunc(Offset.getBitWidth());
    Offset += Index * APInt(Offset.getBitWidth(),
                               DL.getTypeAllocSize(GTI.getIndexedType()));
  }
  return true;
}
