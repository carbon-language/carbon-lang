//===- ConstantFolder.h - Constant folding helper ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstantFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants
// with minimal folding.  For general constant creation and folding,
// use ConstantExpr and the routines in llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONSTANTFOLDER_H
#define LLVM_IR_CONSTANTFOLDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilderFolder.h"

namespace llvm {

/// ConstantFolder - Create constants with minimum, target independent, folding.
class ConstantFolder final : public IRBuilderFolder {
  virtual void anchor();

public:
  explicit ConstantFolder() = default;

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateAdd(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return ConstantExpr::getAdd(LHS, RHS, HasNUW, HasNSW);
  }

  Constant *CreateFAdd(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getFAdd(LHS, RHS);
  }

  Constant *CreateSub(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return ConstantExpr::getSub(LHS, RHS, HasNUW, HasNSW);
  }

  Constant *CreateFSub(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getFSub(LHS, RHS);
  }

  Constant *CreateMul(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return ConstantExpr::getMul(LHS, RHS, HasNUW, HasNSW);
  }

  Constant *CreateFMul(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getFMul(LHS, RHS);
  }

  Constant *CreateUDiv(Constant *LHS, Constant *RHS,
                               bool isExact = false) const override {
    return ConstantExpr::getUDiv(LHS, RHS, isExact);
  }

  Constant *CreateSDiv(Constant *LHS, Constant *RHS,
                               bool isExact = false) const override {
    return ConstantExpr::getSDiv(LHS, RHS, isExact);
  }

  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getFDiv(LHS, RHS);
  }

  Constant *CreateURem(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getURem(LHS, RHS);
  }

  Constant *CreateSRem(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getSRem(LHS, RHS);
  }

  Constant *CreateFRem(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getFRem(LHS, RHS);
  }

  Constant *CreateShl(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return ConstantExpr::getShl(LHS, RHS, HasNUW, HasNSW);
  }

  Constant *CreateLShr(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return ConstantExpr::getLShr(LHS, RHS, isExact);
  }

  Constant *CreateAShr(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return ConstantExpr::getAShr(LHS, RHS, isExact);
  }

  Constant *CreateAnd(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getAnd(LHS, RHS);
  }

  Constant *CreateOr(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getOr(LHS, RHS);
  }

  Constant *CreateXor(Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::getXor(LHS, RHS);
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const override {
    return ConstantExpr::get(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return ConstantExpr::getNeg(C, HasNUW, HasNSW);
  }

  Constant *CreateFNeg(Constant *C) const override {
    return ConstantExpr::getFNeg(C);
  }

  Constant *CreateNot(Constant *C) const override {
    return ConstantExpr::getNot(C);
  }

  Constant *CreateUnOp(Instruction::UnaryOps Opc, Constant *C) const override {
    return ConstantExpr::get(Opc, C);
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Type *Ty, Constant *C,
                                ArrayRef<Constant *> IdxList) const override {
    return ConstantExpr::getGetElementPtr(Ty, C, IdxList);
  }

  Constant *CreateGetElementPtr(Type *Ty, Constant *C,
                                Constant *Idx) const override {
    // This form of the function only exists to avoid ambiguous overload
    // warnings about whether to convert Idx to ArrayRef<Constant *> or
    // ArrayRef<Value *>.
    return ConstantExpr::getGetElementPtr(Ty, C, Idx);
  }

  Constant *CreateGetElementPtr(Type *Ty, Constant *C,
                                ArrayRef<Value *> IdxList) const override {
    return ConstantExpr::getGetElementPtr(Ty, C, IdxList);
  }

  Constant *CreateInBoundsGetElementPtr(
      Type *Ty, Constant *C, ArrayRef<Constant *> IdxList) const override {
    return ConstantExpr::getInBoundsGetElementPtr(Ty, C, IdxList);
  }

  Constant *CreateInBoundsGetElementPtr(Type *Ty, Constant *C,
                                        Constant *Idx) const override {
    // This form of the function only exists to avoid ambiguous overload
    // warnings about whether to convert Idx to ArrayRef<Constant *> or
    // ArrayRef<Value *>.
    return ConstantExpr::getInBoundsGetElementPtr(Ty, C, Idx);
  }

  Constant *CreateInBoundsGetElementPtr(
      Type *Ty, Constant *C, ArrayRef<Value *> IdxList) const override {
    return ConstantExpr::getInBoundsGetElementPtr(Ty, C, IdxList);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       Type *DestTy) const override {
    return ConstantExpr::getCast(Op, C, DestTy);
  }

  Constant *CreatePointerCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getPointerCast(C, DestTy);
  }

  Constant *CreatePointerBitCastOrAddrSpaceCast(Constant *C,
                                                Type *DestTy) const override {
    return ConstantExpr::getPointerBitCastOrAddrSpaceCast(C, DestTy);
  }

  Constant *CreateIntCast(Constant *C, Type *DestTy,
                          bool isSigned) const override {
    return ConstantExpr::getIntegerCast(C, DestTy, isSigned);
  }

  Constant *CreateFPCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getFPCast(C, DestTy);
  }

  Constant *CreateBitCast(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }

  Constant *CreateIntToPtr(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }

  Constant *CreatePtrToInt(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }

  Constant *CreateZExtOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getZExtOrBitCast(C, DestTy);
  }

  Constant *CreateSExtOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getSExtOrBitCast(C, DestTy);
  }

  Constant *CreateTruncOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateICmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const override {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }

  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const override {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateSelect(Constant *C, Constant *True,
                         Constant *False) const override {
    return ConstantExpr::getSelect(C, True, False);
  }

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const override {
    return ConstantExpr::getExtractElement(Vec, Idx);
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const override {
    return ConstantExpr::getInsertElement(Vec, NewElt, Idx);
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                ArrayRef<int> Mask) const override {
    return ConstantExpr::getShuffleVector(V1, V2, Mask);
  }

  Constant *CreateExtractValue(Constant *Agg,
                               ArrayRef<unsigned> IdxList) const override {
    return ConstantExpr::getExtractValue(Agg, IdxList);
  }

  Constant *CreateInsertValue(Constant *Agg, Constant *Val,
                              ArrayRef<unsigned> IdxList) const override {
    return ConstantExpr::getInsertValue(Agg, Val, IdxList);
  }
};

} // end namespace llvm

#endif // LLVM_IR_CONSTANTFOLDER_H
