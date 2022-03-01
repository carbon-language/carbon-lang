//===- IRBuilderFolder.h - Const folder interface for IRBuilder -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines for constant folding interface used by IRBuilder.
// It is implemented by ConstantFolder (default), TargetFolder and NoFoler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_IRBUILDERFOLDER_H
#define LLVM_IR_IRBUILDERFOLDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

/// IRBuilderFolder - Interface for constant folding in IRBuilder.
class IRBuilderFolder {
public:
  virtual ~IRBuilderFolder();

  //===--------------------------------------------------------------------===//
  // Value-based folders.
  //
  // Return an existing value or a constant if the operation can be simplified.
  // Otherwise return nullptr.
  //===--------------------------------------------------------------------===//
  virtual Value *FoldAdd(Value *LHS, Value *RHS, bool HasNUW = false,
                         bool HasNSW = false) const = 0;

  virtual Value *FoldAnd(Value *LHS, Value *RHS) const = 0;

  virtual Value *FoldOr(Value *LHS, Value *RHS) const = 0;

  virtual Value *FoldICmp(CmpInst::Predicate P, Value *LHS,
                          Value *RHS) const = 0;

  virtual Value *FoldGEP(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                         bool IsInBounds = false) const = 0;

  virtual Value *FoldSelect(Value *C, Value *True, Value *False) const = 0;

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  virtual Value *CreateFAdd(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateSub(Constant *LHS, Constant *RHS,
                           bool HasNUW = false, bool HasNSW = false) const = 0;
  virtual Value *CreateFSub(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateMul(Constant *LHS, Constant *RHS,
                           bool HasNUW = false, bool HasNSW = false) const = 0;
  virtual Value *CreateFMul(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateUDiv(Constant *LHS, Constant *RHS,
                            bool isExact = false) const = 0;
  virtual Value *CreateSDiv(Constant *LHS, Constant *RHS,
                            bool isExact = false) const = 0;
  virtual Value *CreateFDiv(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateURem(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateSRem(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateFRem(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateShl(Constant *LHS, Constant *RHS,
                           bool HasNUW = false, bool HasNSW = false) const = 0;
  virtual Value *CreateLShr(Constant *LHS, Constant *RHS,
                            bool isExact = false) const = 0;
  virtual Value *CreateAShr(Constant *LHS, Constant *RHS,
                            bool isExact = false) const = 0;
  virtual Value *CreateXor(Constant *LHS, Constant *RHS) const = 0;
  virtual Value *CreateBinOp(Instruction::BinaryOps Opc,
                             Constant *LHS, Constant *RHS) const = 0;

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  virtual Value *CreateNeg(Constant *C,
                           bool HasNUW = false, bool HasNSW = false) const = 0;
  virtual Value *CreateFNeg(Constant *C) const = 0;
  virtual Value *CreateNot(Constant *C) const = 0;
  virtual Value *CreateUnOp(Instruction::UnaryOps Opc, Constant *C) const = 0;

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  virtual Value *CreateCast(Instruction::CastOps Op, Constant *C,
                            Type *DestTy) const = 0;
  virtual Value *CreatePointerCast(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreatePointerBitCastOrAddrSpaceCast(Constant *C,
                                                     Type *DestTy) const = 0;
  virtual Value *CreateIntCast(Constant *C, Type *DestTy,
                               bool isSigned) const = 0;
  virtual Value *CreateFPCast(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreateBitCast(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreateIntToPtr(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreatePtrToInt(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreateZExtOrBitCast(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreateSExtOrBitCast(Constant *C, Type *DestTy) const = 0;
  virtual Value *CreateTruncOrBitCast(Constant *C, Type *DestTy) const = 0;

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  virtual Value *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                            Constant *RHS) const = 0;

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  virtual Value *CreateExtractElement(Constant *Vec, Constant *Idx) const = 0;
  virtual Value *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                     Constant *Idx) const = 0;
  virtual Value *CreateShuffleVector(Constant *V1, Constant *V2,
                                     ArrayRef<int> Mask) const = 0;
  virtual Value *CreateExtractValue(Constant *Agg,
                                    ArrayRef<unsigned> IdxList) const = 0;
  virtual Value *CreateInsertValue(Constant *Agg, Constant *Val,
                                   ArrayRef<unsigned> IdxList) const = 0;
};

} // end namespace llvm

#endif // LLVM_IR_IRBUILDERFOLDER_H
