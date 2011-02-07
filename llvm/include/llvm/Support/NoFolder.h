//======-- llvm/Support/NoFolder.h - Constant folding helper -*- C++ -*-======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the NoFolder class, a helper for IRBuilder.  It provides
// IRBuilder with a set of methods for creating unfolded constants.  This is
// useful for learners trying to understand how LLVM IR works, and who don't
// want details to be hidden by the constant folder.  For general constant
// creation and folding, use ConstantExpr and the routines in
// llvm/Analysis/ConstantFolding.h.
//
// Note: since it is not actually possible to create unfolded constants, this
// class returns instructions rather than constants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NOFOLDER_H
#define LLVM_SUPPORT_NOFOLDER_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"

namespace llvm {

class LLVMContext;

/// NoFolder - Create "constants" (actually, instructions) with no folding.
class NoFolder {
public:
  explicit NoFolder(LLVMContext &) {}

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateAdd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAdd(LHS, RHS);
  }
  Instruction *CreateNSWAdd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNSWAdd(LHS, RHS);
  }
  Instruction *CreateNUWAdd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNUWAdd(LHS, RHS);
  }
  Instruction *CreateFAdd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFAdd(LHS, RHS);
  }
  Instruction *CreateSub(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSub(LHS, RHS);
  }
  Instruction *CreateNSWSub(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNSWSub(LHS, RHS);
  }
  Instruction *CreateNUWSub(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNUWSub(LHS, RHS);
  }
  Instruction *CreateFSub(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFSub(LHS, RHS);
  }
  Instruction *CreateMul(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateMul(LHS, RHS);
  }
  Instruction *CreateNSWMul(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNSWMul(LHS, RHS);
  }
  Instruction *CreateNUWMul(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateNUWMul(LHS, RHS);
  }
  Instruction *CreateFMul(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFMul(LHS, RHS);
  }
  Instruction *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateUDiv(LHS, RHS);
  }
  Instruction *CreateExactUDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateExactUDiv(LHS, RHS);
  }
  Instruction *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSDiv(LHS, RHS);
  }
  Instruction *CreateExactSDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateExactSDiv(LHS, RHS);
  }
  Instruction *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFDiv(LHS, RHS);
  }
  Instruction *CreateURem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateURem(LHS, RHS);
  }
  Instruction *CreateSRem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSRem(LHS, RHS);
  }
  Instruction *CreateFRem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFRem(LHS, RHS);
  }
  Instruction *CreateShl(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateShl(LHS, RHS);
  }
  Instruction *CreateLShr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateLShr(LHS, RHS);
  }
  Instruction *CreateAShr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAShr(LHS, RHS);
  }
  Instruction *CreateAnd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAnd(LHS, RHS);
  }
  Instruction *CreateOr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateOr(LHS, RHS);
  }
  Instruction *CreateXor(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateXor(LHS, RHS);
  }

  Instruction *CreateBinOp(Instruction::BinaryOps Opc,
                           Constant *LHS, Constant *RHS) const {
    return BinaryOperator::Create(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateNeg(Constant *C) const {
    return BinaryOperator::CreateNeg(C);
  }
  Instruction *CreateNSWNeg(Constant *C) const {
    return BinaryOperator::CreateNSWNeg(C);
  }
  Instruction *CreateNUWNeg(Constant *C) const {
    return BinaryOperator::CreateNUWNeg(C);
  }
  Instruction *CreateFNeg(Constant *C) const {
    return BinaryOperator::CreateFNeg(C);
  }
  Instruction *CreateNot(Constant *C) const {
    return BinaryOperator::CreateNot(C);
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
  }
  Instruction *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                                   unsigned NumIdx) const {
    return GetElementPtrInst::Create(C, IdxList, IdxList+NumIdx);
  }

  Constant *CreateInBoundsGetElementPtr(Constant *C, Constant* const *IdxList,
                                        unsigned NumIdx) const {
    return ConstantExpr::getInBoundsGetElementPtr(C, IdxList, NumIdx);
  }
  Instruction *CreateInBoundsGetElementPtr(Constant *C, Value* const *IdxList,
                                           unsigned NumIdx) const {
    return GetElementPtrInst::CreateInBounds(C, IdxList, IdxList+NumIdx);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateCast(Instruction::CastOps Op, Constant *C,
                    const Type *DestTy) const {
    return CastInst::Create(Op, C, DestTy);
  }
  Instruction *CreatePointerCast(Constant *C, const Type *DestTy) const {
    return CastInst::CreatePointerCast(C, DestTy);
  }
  Instruction *CreateIntCast(Constant *C, const Type *DestTy,
                       bool isSigned) const {
    return CastInst::CreateIntegerCast(C, DestTy, isSigned);
  }
  Instruction *CreateFPCast(Constant *C, const Type *DestTy) const {
    return CastInst::CreateFPCast(C, DestTy);
  }

  Instruction *CreateBitCast(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }
  Instruction *CreateIntToPtr(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }
  Instruction *CreatePtrToInt(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }
  Instruction *CreateZExtOrBitCast(Constant *C, const Type *DestTy) const {
    return CastInst::CreateZExtOrBitCast(C, DestTy);
  }
  Instruction *CreateSExtOrBitCast(Constant *C, const Type *DestTy) const {
    return CastInst::CreateSExtOrBitCast(C, DestTy);
  }

  Instruction *CreateTruncOrBitCast(Constant *C, const Type *DestTy) const {
    return CastInst::CreateTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Instruction *CreateICmp(CmpInst::Predicate P,
                          Constant *LHS, Constant *RHS) const {
    return new ICmpInst(P, LHS, RHS);
  }
  Instruction *CreateFCmp(CmpInst::Predicate P,
                          Constant *LHS, Constant *RHS) const {
    return new FCmpInst(P, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Instruction *CreateSelect(Constant *C,
                            Constant *True, Constant *False) const {
    return SelectInst::Create(C, True, False);
  }

  Instruction *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return ExtractElementInst::Create(Vec, Idx);
  }

  Instruction *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                   Constant *Idx) const {
    return InsertElementInst::Create(Vec, NewElt, Idx);
  }

  Instruction *CreateShuffleVector(Constant *V1, Constant *V2,
                                   Constant *Mask) const {
    return new ShuffleVectorInst(V1, V2, Mask);
  }

  Instruction *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                                  unsigned NumIdx) const {
    return ExtractValueInst::Create(Agg, IdxList, IdxList+NumIdx);
  }

  Instruction *CreateInsertValue(Constant *Agg, Constant *Val,
                                 const unsigned *IdxList,
                                 unsigned NumIdx) const {
    return InsertValueInst::Create(Agg, Val, IdxList, IdxList+NumIdx);
  }
};

}

#endif
