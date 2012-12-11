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

  return GEPI.accumulateConstantOffset(DL, Offset);
}
