//===- InstCombineAtomicRMW.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for atomic rmw instructions.
//
//===----------------------------------------------------------------------===//
#include "InstCombineInternal.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace {
/// Return true if and only if the given instruction does not modify the memory
/// location referenced.  Note that an idemptent atomicrmw may still have
/// ordering effects on nearby instructions, or be volatile.
/// TODO: Common w/ the version in AtomicExpandPass, and change the term used.
/// Idemptotent is confusing in this context.
bool isIdempotentRMW(AtomicRMWInst& RMWI) {
  auto C = dyn_cast<ConstantInt>(RMWI.getValOperand());
  if(!C)
    // TODO: Handle fadd, fsub?
    return false;

  AtomicRMWInst::BinOp Op = RMWI.getOperation();
  switch(Op) {
    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor:
      return C->isZero();
    case AtomicRMWInst::And:
      return C->isMinusOne();
    case AtomicRMWInst::Min:
      return C->isMaxValue(true);
    case AtomicRMWInst::Max:
      return C->isMinValue(true);
    case AtomicRMWInst::UMin:
      return C->isMaxValue(false);
    case AtomicRMWInst::UMax:
      return C->isMinValue(false);
    default:
      return false;
  }
}
}


Instruction *InstCombiner::visitAtomicRMWInst(AtomicRMWInst &RMWI) {
  if (!isIdempotentRMW(RMWI))
    return nullptr;

  // TODO: Canonicalize the operation for an idempotent operation we can't
  // convert into a simple load.

  // Volatile RMWs perform a load and a store, we cannot replace
  // this by just a load.
  if (RMWI.isVolatile())
    return nullptr;

  // Check if the required ordering is compatible with an atomic load.
  AtomicOrdering Ordering = RMWI.getOrdering();
  assert(Ordering != AtomicOrdering::NotAtomic &&
         Ordering != AtomicOrdering::Unordered &&
         "AtomicRMWs don't make sense with Unordered or NotAtomic");
  if (Ordering != AtomicOrdering::Acquire &&
      Ordering != AtomicOrdering::Monotonic)
    return nullptr;
  
  LoadInst *Load = new LoadInst(RMWI.getType(), RMWI.getPointerOperand());
  Load->setAtomic(Ordering, RMWI.getSyncScopeID());
  Load->setAlignment(DL.getABITypeAlignment(RMWI.getType()));
  return Load;
}
