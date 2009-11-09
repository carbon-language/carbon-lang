//===- InstructionSimplify.cpp - Fold instruction operands ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for folding instructions into simpler forms
// that do not require creating new instructions.  For example, this does
// constant folding, and can handle identities like (X&0)->0.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Instructions.h"
using namespace llvm;


/// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS, 
                           const TargetData *TD) {
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS)) {
      Constant *COps[] = {CLHS, CRHS};
      return ConstantFoldInstOperands(Opcode, LHS->getType(), COps, 2, TD);
    }     
  return 0;
}


/// SimplifyCompare - Given operands for a CmpInst, see if we can
/// fold the result.
Value *llvm::SimplifyCompare(unsigned Predicate, Value *LHS, Value *RHS,
                             const TargetData *TD) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, TD);
  
  // If this is an integer compare and the LHS and RHS are the same, fold it.
  if (LHS == RHS)
    if (isa<IntegerType>(LHS->getType()) || isa<PointerType>(LHS->getType())) {
      if (ICmpInst::isTrueWhenEqual(Pred))
        return ConstantInt::getTrue(LHS->getContext());
      else
        return ConstantInt::getFalse(LHS->getContext());
    }
  return 0;
}

