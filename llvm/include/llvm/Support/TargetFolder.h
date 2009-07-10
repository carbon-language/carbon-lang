//====-- llvm/Support/TargetFolder.h - Constant folding helper -*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants with
// target dependent folding, in addition to the same target-independent
// folding that the ConstantFolder class provides.  For general constant
// creation and folding, use ConstantExpr and the routines in
// llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETFOLDER_H
#define LLVM_SUPPORT_TARGETFOLDER_H

#include "llvm/Constants.h"
#include "llvm/Analysis/ConstantFolding.h"

namespace llvm {

class TargetData;
class LLVMContext;

/// TargetFolder - Create constants with target dependent folding.
class TargetFolder {
  const TargetData *TD;
  LLVMContext &Context;

  /// Fold - Fold the constant using target specific information.
  Constant *Fold(Constant *C) const {
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
      if (Constant *CF = ConstantFoldConstantExpression(CE, &Context, TD))
        return CF;
    return C;
  }

public:
  explicit TargetFolder(const TargetData *TheTD, LLVMContext &C) :
    TD(TheTD), Context(C) {}

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateAdd(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprAdd(LHS, RHS));
  }
  Constant *CreateFAdd(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprFAdd(LHS, RHS));
  }
  Constant *CreateSub(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprSub(LHS, RHS));
  }
  Constant *CreateFSub(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprFSub(LHS, RHS));
  }
  Constant *CreateMul(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprMul(LHS, RHS));
  }
  Constant *CreateFMul(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprFMul(LHS, RHS));
  }
  Constant *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprUDiv(LHS, RHS));
  }
  Constant *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprSDiv(LHS, RHS));
  }
  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprFDiv(LHS, RHS));
  }
  Constant *CreateURem(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprURem(LHS, RHS));
  }
  Constant *CreateSRem(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprSRem(LHS, RHS));
  }
  Constant *CreateFRem(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprFRem(LHS, RHS));
  }
  Constant *CreateShl(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprShl(LHS, RHS));
  }
  Constant *CreateLShr(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprLShr(LHS, RHS));
  }
  Constant *CreateAShr(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprAShr(LHS, RHS));
  }
  Constant *CreateAnd(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprAnd(LHS, RHS));
  }
  Constant *CreateOr(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprOr(LHS, RHS));
  }
  Constant *CreateXor(Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExprXor(LHS, RHS));
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const {
    return Fold(Context.getConstantExpr(Opc, LHS, RHS));
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C) const {
    return Fold(Context.getConstantExprNeg(C));
  }
  Constant *CreateFNeg(Constant *C) const {
    return Fold(Context.getConstantExprFNeg(C));
  }
  Constant *CreateNot(Constant *C) const {
    return Fold(Context.getConstantExprNot(C));
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return Fold(Context.getConstantExprGetElementPtr(C, IdxList, NumIdx));
  }
  Constant *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                                unsigned NumIdx) const {
    return Fold(Context.getConstantExprGetElementPtr(C, IdxList, NumIdx));
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       const Type *DestTy) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(Context.getConstantExprCast(Op, C, DestTy));
  }
  Constant *CreateIntCast(Constant *C, const Type *DestTy,
                          bool isSigned) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(Context.getConstantExprIntegerCast(C, DestTy, isSigned));
  }

  Constant *CreateBitCast(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }
  Constant *CreateIntToPtr(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }
  Constant *CreatePtrToInt(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }
  Constant *CreateTruncOrBitCast(Constant *C, const Type *DestTy) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(Context.getConstantExprTruncOrBitCast(C, DestTy));
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateICmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Fold(Context.getConstantExprCompare(P, LHS, RHS));
  }
  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Fold(Context.getConstantExprCompare(P, LHS, RHS));
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateSelect(Constant *C, Constant *True, Constant *False) const {
    return Fold(Context.getConstantExprSelect(C, True, False));
  }

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return Fold(Context.getConstantExprExtractElement(Vec, Idx));
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const {
    return Fold(Context.getConstantExprInsertElement(Vec, NewElt, Idx));
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                Constant *Mask) const {
    return Fold(Context.getConstantExprShuffleVector(V1, V2, Mask));
  }

  Constant *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                               unsigned NumIdx) const {
    return Fold(Context.getConstantExprExtractValue(Agg, IdxList, NumIdx));
  }

  Constant *CreateInsertValue(Constant *Agg, Constant *Val,
                              const unsigned *IdxList, unsigned NumIdx) const {
    return Fold(Context.getConstantExprInsertValue(Agg, Val, IdxList, NumIdx));
  }
};

}

#endif
