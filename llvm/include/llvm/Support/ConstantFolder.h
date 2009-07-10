//===-- llvm/Support/ConstantFolder.h - Constant folding helper -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstantFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants
// with minimal folding.  For general constant creation and folding,
// use ConstantExpr and the routines in llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CONSTANTFOLDER_H
#define LLVM_SUPPORT_CONSTANTFOLDER_H

#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"

namespace llvm {
  
/// ConstantFolder - Create constants with minimum, target independent, folding.
class ConstantFolder {
  LLVMContext &Context;
  
public:
  ConstantFolder(LLVMContext &C) : Context(C) { }

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateAdd(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprAdd(LHS, RHS);
  }
  Constant *CreateFAdd(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprFAdd(LHS, RHS);
  }
  Constant *CreateSub(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprSub(LHS, RHS);
  }
  Constant *CreateFSub(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprFSub(LHS, RHS);
  }
  Constant *CreateMul(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprMul(LHS, RHS);
  }
  Constant *CreateFMul(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprFMul(LHS, RHS);
  }
  Constant *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprUDiv(LHS, RHS);
  }
  Constant *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprSDiv(LHS, RHS);
  }
  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprFDiv(LHS, RHS);
  }
  Constant *CreateURem(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprURem(LHS, RHS);
  }
  Constant *CreateSRem(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprSRem(LHS, RHS);
  }
  Constant *CreateFRem(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprFRem(LHS, RHS);
  }
  Constant *CreateShl(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprShl(LHS, RHS);
  }
  Constant *CreateLShr(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprLShr(LHS, RHS);
  }
  Constant *CreateAShr(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprAShr(LHS, RHS);
  }
  Constant *CreateAnd(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprAnd(LHS, RHS);
  }
  Constant *CreateOr(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprOr(LHS, RHS);
  }
  Constant *CreateXor(Constant *LHS, Constant *RHS) const {
    return Context.getConstantExprXor(LHS, RHS);
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const {
    return Context.getConstantExpr(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C) const {
    return Context.getConstantExprNeg(C);
  }
  Constant *CreateFNeg(Constant *C) const {
    return Context.getConstantExprFNeg(C);
  }
  Constant *CreateNot(Constant *C) const {
    return Context.getConstantExprNot(C);
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return Context.getConstantExprGetElementPtr(C, IdxList, NumIdx);
  }
  Constant *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                                unsigned NumIdx) const {
    return Context.getConstantExprGetElementPtr(C, IdxList, NumIdx);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       const Type *DestTy) const {
    return Context.getConstantExprCast(Op, C, DestTy);
  }
  Constant *CreateIntCast(Constant *C, const Type *DestTy,
                          bool isSigned) const {
    return Context.getConstantExprIntegerCast(C, DestTy, isSigned);
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
    return Context.getConstantExprTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateICmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Context.getConstantExprCompare(P, LHS, RHS);
  }
  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Context.getConstantExprCompare(P, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateSelect(Constant *C, Constant *True, Constant *False) const {
    return Context.getConstantExprSelect(C, True, False);
  }

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return Context.getConstantExprExtractElement(Vec, Idx);
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const {
    return Context.getConstantExprInsertElement(Vec, NewElt, Idx);
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                Constant *Mask) const {
    return Context.getConstantExprShuffleVector(V1, V2, Mask);
  }

  Constant *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                               unsigned NumIdx) const {
    return Context.getConstantExprExtractValue(Agg, IdxList, NumIdx);
  }

  Constant *CreateInsertValue(Constant *Agg, Constant *Val,
                              const unsigned *IdxList, unsigned NumIdx) const {
    return Context.getConstantExprInsertValue(Agg, Val, IdxList, NumIdx);
  }
};

}

#endif
