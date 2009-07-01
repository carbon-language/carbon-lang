//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
// class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

const LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl()) { }
LLVMContext::~LLVMContext() { delete pImpl; }

// ConstantInt accessors.
ConstantInt* LLVMContext::getConstantIntTrue() const {
  return ConstantInt::getTrue();
}

ConstantInt* LLVMContext::getConstantIntFalse() const {
  return ConstantInt::getFalse();
}

ConstantInt* LLVMContext::getConstantInt(const IntegerType* Ty, uint64_t V,
                                         bool isSigned) const {
  return ConstantInt::get(Ty, V, isSigned);
}

ConstantInt* LLVMContext::getConstantIntSigned(const IntegerType* Ty,
                                               int64_t V) const {
  return ConstantInt::getSigned(Ty, V);
}

ConstantInt* LLVMContext::getConstantInt(const APInt& V) const {
  return ConstantInt::get(V);
}

Constant* LLVMContext::getConstantInt(const Type* Ty, const APInt& V) const {
  return ConstantInt::get(Ty, V);
}

ConstantInt* LLVMContext::getAllOnesConstantInt(const Type* Ty) const {
  return ConstantInt::getAllOnesValue(Ty);
}


// ConstantPointerNull accessors.
ConstantPointerNull*
LLVMContext::getConstantPointerNull(const PointerType* T) const {
  return ConstantPointerNull::get(T);
}


// ConstantStruct accessors.
Constant* LLVMContext::getConstantStruct(const StructType* T,
                                        const std::vector<Constant*>& V) const {
  return ConstantStruct::get(T, V);
}

Constant* LLVMContext::getConstantStruct(const std::vector<Constant*>& V,
                                         bool Packed) const {
  return ConstantStruct::get(V, Packed);
}

Constant* LLVMContext::getConstantStruct(Constant* const *Vals,
                                         unsigned NumVals, bool Packed) const {
  return ConstantStruct::get(Vals, NumVals, Packed);
}


// ConstantAggregateZero accessors.
ConstantAggregateZero*
LLVMContext::getConstantAggregateZero(const Type* Ty) const {
  return ConstantAggregateZero::get(Ty);
}


// ConstantArray accessors.
Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        const std::vector<Constant*>& V) const {
  return ConstantArray::get(T, V);
}

Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        Constant* const* Vals,
                                        unsigned NumVals) const {
  return ConstantArray::get(T, Vals, NumVals);
}

Constant* LLVMContext::getConstantArray(const std::string& Initializer,
                                        bool AddNull) const {
  return ConstantArray::get(Initializer, AddNull);
}


// ConstantExpr accessors.
Constant* LLVMContext::getConstantExpr(unsigned Opcode, Constant* C1,
                                       Constant* C2) const {
  return ConstantExpr::get(Opcode, C1, C2);
}

Constant* LLVMContext::getConstantExprTrunc(Constant* C, const Type* Ty) const {
  return ConstantExpr::getTrunc(C, Ty);
}

Constant* LLVMContext::getConstantExprSExt(Constant* C, const Type* Ty) const {
  return ConstantExpr::getSExt(C, Ty);
}

Constant* LLVMContext::getConstantExprZExt(Constant* C, const Type* Ty) const {
  return ConstantExpr::getZExt(C, Ty);  
}

Constant* 
LLVMContext::getConstantExprFPTrunc(Constant* C, const Type* Ty) const {
  return ConstantExpr::getFPTrunc(C, Ty);
}

Constant*
LLVMContext::getConstantExprFPExtend(Constant* C, const Type* Ty) const {
  return ConstantExpr::getFPExtend(C, Ty);
}

Constant* 
LLVMContext::getConstantExprUIToFP(Constant* C, const Type* Ty) const {
  return ConstantExpr::getUIToFP(C, Ty);
}

Constant*
LLVMContext::getConstantExprSIToFP(Constant* C, const Type* Ty) const {
  return ConstantExpr::getSIToFP(C, Ty);
}

Constant*
LLVMContext::getConstantExprFPToUI(Constant* C, const Type* Ty) const {
  return ConstantExpr::getFPToUI(C, Ty);
}

Constant*
LLVMContext::getConstantExprFPToSI(Constant* C, const Type* Ty) const {
  return ConstantExpr::getFPToSI(C, Ty);
}

Constant*
LLVMContext::getConstantExprPtrToInt(Constant* C, const Type* Ty) const {
  return ConstantExpr::getPtrToInt(C, Ty);
}

Constant*
LLVMContext::getConstantExprIntToPtr(Constant* C, const Type* Ty) const {
  return ConstantExpr::getIntToPtr(C, Ty);
}

Constant*
LLVMContext::getConstantExprBitCast(Constant* C, const Type* Ty) const {
  return ConstantExpr::getBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprCast(unsigned ops, Constant* C,
                                           const Type* Ty) const {
  return ConstantExpr::getCast(ops, C, Ty);
}

Constant* LLVMContext::getConstantExprZExtOrBitCast(Constant* C,
                                                    const Type* Ty) const {
  return ConstantExpr::getZExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSExtOrBitCast(Constant* C,
                                                    const Type* Ty) const {
  return ConstantExpr::getSExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprTruncOrBitCast(Constant* C,
                                                     const Type* Ty) const {
  return ConstantExpr::getTruncOrBitCast(C, Ty);  
}

Constant*
LLVMContext::getConstantExprPointerCast(Constant* C, const Type* Ty) const {
  return ConstantExpr::getPointerCast(C, Ty);
}

Constant* LLVMContext::getConstantExprIntegerCast(Constant* C, const Type* Ty,
                                                  bool isSigned) const {
  return ConstantExpr::getIntegerCast(C, Ty, isSigned);
}

Constant*
LLVMContext::getConstantExprFPCast(Constant* C, const Type* Ty) const {
  return ConstantExpr::getFPCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSelect(Constant* C, Constant* V1,
                                             Constant* V2) const {
  return ConstantExpr::getSelect(C, V1, V2);
}

Constant* LLVMContext::getConstantExprAlignOf(const Type* Ty) const {
  return ConstantExpr::getAlignOf(Ty);
}

Constant* LLVMContext::getConstantExprCompare(unsigned short pred,
                                 Constant* C1, Constant* C2) const {
  return ConstantExpr::getCompare(pred, C1, C2);
}

Constant* LLVMContext::getConstantExprNeg(Constant* C) const {
  return ConstantExpr::getNeg(C);
}

Constant* LLVMContext::getConstantExprFNeg(Constant* C) const {
  return ConstantExpr::getFNeg(C);
}

Constant* LLVMContext::getConstantExprNot(Constant* C) const {
  return ConstantExpr::getNot(C);
}

Constant* LLVMContext::getConstantExprAdd(Constant* C1, Constant* C2) const {
  return ConstantExpr::getAdd(C1, C2);
}

Constant* LLVMContext::getConstantExprFAdd(Constant* C1, Constant* C2) const {
  return ConstantExpr::getFAdd(C1, C2);
}

Constant* LLVMContext::getConstantExprSub(Constant* C1, Constant* C2) const {
  return ConstantExpr::getSub(C1, C2);
}

Constant* LLVMContext::getConstantExprFSub(Constant* C1, Constant* C2) const {
  return ConstantExpr::getFSub(C1, C2);
}

Constant* LLVMContext::getConstantExprMul(Constant* C1, Constant* C2) const {
  return ConstantExpr::getMul(C1, C2);
}

Constant* LLVMContext::getConstantExprFMul(Constant* C1, Constant* C2) const {
  return ConstantExpr::getFMul(C1, C2);
}

Constant* LLVMContext::getConstantExprUDiv(Constant* C1, Constant* C2) const {
  return ConstantExpr::getUDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprSDiv(Constant* C1, Constant* C2) const {
  return ConstantExpr::getSDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprFDiv(Constant* C1, Constant* C2) const {
  return ConstantExpr::getFDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprURem(Constant* C1, Constant* C2) const {
  return ConstantExpr::getURem(C1, C2);
}

Constant* LLVMContext::getConstantExprSRem(Constant* C1, Constant* C2) const {
  return ConstantExpr::getSRem(C1, C2);
}

Constant* LLVMContext::getConstantExprFRem(Constant* C1, Constant* C2) const {
  return ConstantExpr::getFRem(C1, C2);
}

Constant* LLVMContext::getConstantExprAnd(Constant* C1, Constant* C2) const {
  return ConstantExpr::getAnd(C1, C2);
}

Constant* LLVMContext::getConstantExprOr(Constant* C1, Constant* C2) const {
  return ConstantExpr::getOr(C1, C2);
}

Constant* LLVMContext::getConstantExprXor(Constant* C1, Constant* C2) const {
  return ConstantExpr::getXor(C1, C2);
}

Constant* LLVMContext::getConstantExprICmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) const {
  return ConstantExpr::getICmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprFCmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) const {
  return ConstantExpr::getFCmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprVICmp(unsigned short pred, Constant* LHS,
                               Constant* RHS) const {
  return ConstantExpr::getVICmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprVFCmp(unsigned short pred, Constant* LHS,
                               Constant* RHS) const {
  return ConstantExpr::getVFCmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprShl(Constant* C1, Constant* C2) const {
  return ConstantExpr::getShl(C1, C2);
}

Constant* LLVMContext::getConstantExprLShr(Constant* C1, Constant* C2) const {
  return ConstantExpr::getLShr(C1, C2);
}

Constant* LLVMContext::getConstantExprAShr(Constant* C1, Constant* C2) const {
  return ConstantExpr::getAShr(C1, C2);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Constant* const* IdxList, 
                                                    unsigned NumIdx) const {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Value* const* IdxList, 
                                                    unsigned NumIdx) const {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprExtractElement(Constant* Vec,
                                                     Constant* Idx) const {
  return ConstantExpr::getExtractElement(Vec, Idx);
}

Constant* LLVMContext::getConstantExprInsertElement(Constant* Vec,
                                                    Constant* Elt,
                                                    Constant* Idx) const {
  return ConstantExpr::getInsertElement(Vec, Elt, Idx);
}

Constant* LLVMContext::getConstantExprShuffleVector(Constant* V1, Constant* V2,
                                                    Constant* Mask) const {
  return ConstantExpr::getShuffleVector(V1, V2, Mask);
}

Constant* LLVMContext::getConstantExprExtractValue(Constant* Agg,
                                                   const unsigned* IdxList, 
                                                   unsigned NumIdx) const {
  return ConstantExpr::getExtractValue(Agg, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprInsertValue(Constant* Agg, Constant* Val,
                                                  const unsigned* IdxList,
                                                  unsigned NumIdx) const {
  return ConstantExpr::getInsertValue(Agg, Val, IdxList, NumIdx);
}

Constant* LLVMContext::getZeroValueForNegation(const Type* Ty) const {
  return ConstantExpr::getZeroValueForNegationExpr(Ty);
}


// ConstantFP accessors.
ConstantFP* LLVMContext::getConstantFP(const APFloat& V) const {
  return ConstantFP::get(V);
}

Constant* LLVMContext::getConstantFP(const Type* Ty, double V) const {
  return ConstantFP::get(Ty, V);
}

ConstantFP* LLVMContext::getConstantFPNegativeZero(const Type* Ty) const {
  return ConstantFP::getNegativeZero(Ty);
}


// ConstantVector accessors.
Constant* LLVMContext::getConstantVector(const VectorType* T,
                            const std::vector<Constant*>& V) const {
  return ConstantVector::get(T, V);
}

Constant*
LLVMContext::getConstantVector(const std::vector<Constant*>& V) const {
  return ConstantVector::get(V);
}

Constant* LLVMContext::getConstantVector(Constant* const* Vals,
                                         unsigned NumVals) const {
  return ConstantVector::get(Vals, NumVals);
}

ConstantVector*
LLVMContext::getConstantVectorAllOnes(const VectorType* Ty) const {
  return ConstantVector::getAllOnesValue(Ty);
}

// FunctionType accessors
FunctionType* LLVMContext::getFunctionType(const Type* Result,
                                         const std::vector<const Type*>& Params,
                                         bool isVarArg) const {
  return FunctionType::get(Result, Params, isVarArg);
}
                                
// IntegerType accessors
const IntegerType* LLVMContext::getIntegerType(unsigned NumBits) const {
  return IntegerType::get(NumBits);
}
  
// OpaqueType accessors
OpaqueType* LLVMContext::getOpaqueType() const {
  return OpaqueType::get();
}

// StructType accessors
StructType* LLVMContext::getStructType(const std::vector<const Type*>& Params,
                                       bool isPacked) const {
  return StructType::get(Params, isPacked);
}

// ArrayType accessors
ArrayType* LLVMContext::getArrayType(const Type* ElementType,
                                     uint64_t NumElements) const {
  return ArrayType::get(ElementType, NumElements);
}
  
// PointerType accessors
PointerType* LLVMContext::getPointerType(const Type* ElementType,
                                         unsigned AddressSpace) const {
  return PointerType::get(ElementType, AddressSpace);
}

PointerType*
LLVMContext::getPointerTypeUnqualified(const Type* ElementType) const {
  return PointerType::getUnqual(ElementType);
}
  
// VectorType accessors
VectorType* LLVMContext::getVectorType(const Type* ElementType,
                                       unsigned NumElements) const {
  return VectorType::get(ElementType, NumElements);
}

VectorType* LLVMContext::getVectorTypeInteger(const VectorType* VTy) const {
  return VectorType::getInteger(VTy);  
}

VectorType*
LLVMContext::getVectorTypeExtendedElement(const VectorType* VTy) const {
  return VectorType::getExtendedElementVectorType(VTy);
}

VectorType*
LLVMContext::getVectorTypeTruncatedElement(const VectorType* VTy) const {
  return VectorType::getTruncatedElementVectorType(VTy);
}
