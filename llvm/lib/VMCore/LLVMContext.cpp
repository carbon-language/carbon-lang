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
#include "llvm/MDNode.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl()) { }
LLVMContext::~LLVMContext() { delete pImpl; }

// Constant accessors
Constant* LLVMContext::getNullValue(const Type* Ty) {
  return Constant::getNullValue(Ty);
}

Constant* LLVMContext::getAllOnesValue(const Type* Ty) {
  return Constant::getAllOnesValue(Ty);
}

// UndefValue accessors.
UndefValue* LLVMContext::getUndef(const Type* Ty) {
  return UndefValue::get(Ty);
}

// ConstantInt accessors.
ConstantInt* LLVMContext::getConstantIntTrue() {
  return ConstantInt::getTrue();
}

ConstantInt* LLVMContext::getConstantIntFalse() {
  return ConstantInt::getFalse();
}

ConstantInt* LLVMContext::getConstantInt(const IntegerType* Ty, uint64_t V,
                                         bool isSigned) {
  return ConstantInt::get(Ty, V, isSigned);
}

ConstantInt* LLVMContext::getConstantIntSigned(const IntegerType* Ty,
                                               int64_t V) {
  return ConstantInt::getSigned(Ty, V);
}

ConstantInt* LLVMContext::getConstantInt(const APInt& V) {
  return ConstantInt::get(V);
}

Constant* LLVMContext::getConstantInt(const Type* Ty, const APInt& V) {
  return ConstantInt::get(Ty, V);
}

ConstantInt* LLVMContext::getAllOnesConstantInt(const Type* Ty) {
  return ConstantInt::getAllOnesValue(Ty);
}


// ConstantPointerNull accessors.
ConstantPointerNull* LLVMContext::getConstantPointerNull(const PointerType* T) {
  return ConstantPointerNull::get(T);
}


// ConstantStruct accessors.
Constant* LLVMContext::getConstantStruct(const StructType* T,
                                         const std::vector<Constant*>& V) {
  return ConstantStruct::get(T, V);
}

Constant* LLVMContext::getConstantStruct(const std::vector<Constant*>& V,
                                         bool Packed) {
  return ConstantStruct::get(V, Packed);
}

Constant* LLVMContext::getConstantStruct(Constant* const *Vals,
                                         unsigned NumVals, bool Packed) {
  return ConstantStruct::get(Vals, NumVals, Packed);
}


// ConstantAggregateZero accessors.
ConstantAggregateZero* LLVMContext::getConstantAggregateZero(const Type* Ty) {
  return ConstantAggregateZero::get(Ty);
}


// ConstantArray accessors.
Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        const std::vector<Constant*>& V) {
  return ConstantArray::get(T, V);
}

Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        Constant* const* Vals,
                                        unsigned NumVals) {
  return ConstantArray::get(T, Vals, NumVals);
}

Constant* LLVMContext::getConstantArray(const std::string& Initializer,
                                        bool AddNull) {
  return ConstantArray::get(Initializer, AddNull);
}


// ConstantExpr accessors.
Constant* LLVMContext::getConstantExpr(unsigned Opcode, Constant* C1,
                                       Constant* C2) {
  return ConstantExpr::get(Opcode, C1, C2);
}

Constant* LLVMContext::getConstantExprTrunc(Constant* C, const Type* Ty) {
  return ConstantExpr::getTrunc(C, Ty);
}

Constant* LLVMContext::getConstantExprSExt(Constant* C, const Type* Ty) {
  return ConstantExpr::getSExt(C, Ty);
}

Constant* LLVMContext::getConstantExprZExt(Constant* C, const Type* Ty) {
  return ConstantExpr::getZExt(C, Ty);  
}

Constant* LLVMContext::getConstantExprFPTrunc(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPTrunc(C, Ty);
}

Constant* LLVMContext::getConstantExprFPExtend(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPExtend(C, Ty);
}

Constant* LLVMContext::getConstantExprUIToFP(Constant* C, const Type* Ty) {
  return ConstantExpr::getUIToFP(C, Ty);
}

Constant* LLVMContext::getConstantExprSIToFP(Constant* C, const Type* Ty) {
  return ConstantExpr::getSIToFP(C, Ty);
}

Constant* LLVMContext::getConstantExprFPToUI(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPToUI(C, Ty);
}

Constant* LLVMContext::getConstantExprFPToSI(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPToSI(C, Ty);
}

Constant* LLVMContext::getConstantExprPtrToInt(Constant* C, const Type* Ty) {
  return ConstantExpr::getPtrToInt(C, Ty);
}

Constant* LLVMContext::getConstantExprIntToPtr(Constant* C, const Type* Ty) {
  return ConstantExpr::getIntToPtr(C, Ty);
}

Constant* LLVMContext::getConstantExprBitCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprCast(unsigned ops, Constant* C,
                                           const Type* Ty) {
  return ConstantExpr::getCast(ops, C, Ty);
}

Constant* LLVMContext::getConstantExprZExtOrBitCast(Constant* C,
                                                    const Type* Ty) {
  return ConstantExpr::getZExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSExtOrBitCast(Constant* C,
                                                    const Type* Ty) {
  return ConstantExpr::getSExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprTruncOrBitCast(Constant* C,
                                                     const Type* Ty) {
  return ConstantExpr::getTruncOrBitCast(C, Ty);  
}

Constant* LLVMContext::getConstantExprPointerCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getPointerCast(C, Ty);
}

Constant* LLVMContext::getConstantExprIntegerCast(Constant* C, const Type* Ty,
                                                  bool isSigned) {
  return ConstantExpr::getIntegerCast(C, Ty, isSigned);
}

Constant* LLVMContext::getConstantExprFPCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSelect(Constant* C, Constant* V1,
                                             Constant* V2) {
  return ConstantExpr::getSelect(C, V1, V2);
}

Constant* LLVMContext::getConstantExprAlignOf(const Type* Ty) {
  return ConstantExpr::getAlignOf(Ty);
}

Constant* LLVMContext::getConstantExprCompare(unsigned short pred,
                                 Constant* C1, Constant* C2) {
  return ConstantExpr::getCompare(pred, C1, C2);
}

Constant* LLVMContext::getConstantExprNeg(Constant* C) {
  return ConstantExpr::getNeg(C);
}

Constant* LLVMContext::getConstantExprFNeg(Constant* C) {
  return ConstantExpr::getFNeg(C);
}

Constant* LLVMContext::getConstantExprNot(Constant* C) {
  return ConstantExpr::getNot(C);
}

Constant* LLVMContext::getConstantExprAdd(Constant* C1, Constant* C2) {
  return ConstantExpr::getAdd(C1, C2);
}

Constant* LLVMContext::getConstantExprFAdd(Constant* C1, Constant* C2) {
  return ConstantExpr::getFAdd(C1, C2);
}

Constant* LLVMContext::getConstantExprSub(Constant* C1, Constant* C2) {
  return ConstantExpr::getSub(C1, C2);
}

Constant* LLVMContext::getConstantExprFSub(Constant* C1, Constant* C2) {
  return ConstantExpr::getFSub(C1, C2);
}

Constant* LLVMContext::getConstantExprMul(Constant* C1, Constant* C2) {
  return ConstantExpr::getMul(C1, C2);
}

Constant* LLVMContext::getConstantExprFMul(Constant* C1, Constant* C2) {
  return ConstantExpr::getFMul(C1, C2);
}

Constant* LLVMContext::getConstantExprUDiv(Constant* C1, Constant* C2) {
  return ConstantExpr::getUDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprSDiv(Constant* C1, Constant* C2) {
  return ConstantExpr::getSDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprFDiv(Constant* C1, Constant* C2) {
  return ConstantExpr::getFDiv(C1, C2);
}

Constant* LLVMContext::getConstantExprURem(Constant* C1, Constant* C2) {
  return ConstantExpr::getURem(C1, C2);
}

Constant* LLVMContext::getConstantExprSRem(Constant* C1, Constant* C2) {
  return ConstantExpr::getSRem(C1, C2);
}

Constant* LLVMContext::getConstantExprFRem(Constant* C1, Constant* C2) {
  return ConstantExpr::getFRem(C1, C2);
}

Constant* LLVMContext::getConstantExprAnd(Constant* C1, Constant* C2) {
  return ConstantExpr::getAnd(C1, C2);
}

Constant* LLVMContext::getConstantExprOr(Constant* C1, Constant* C2) {
  return ConstantExpr::getOr(C1, C2);
}

Constant* LLVMContext::getConstantExprXor(Constant* C1, Constant* C2) {
  return ConstantExpr::getXor(C1, C2);
}

Constant* LLVMContext::getConstantExprICmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) {
  return ConstantExpr::getICmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprFCmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) {
  return ConstantExpr::getFCmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprVICmp(unsigned short pred, Constant* LHS,
                               Constant* RHS) {
  return ConstantExpr::getVICmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprVFCmp(unsigned short pred, Constant* LHS,
                               Constant* RHS) {
  return ConstantExpr::getVFCmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprShl(Constant* C1, Constant* C2) {
  return ConstantExpr::getShl(C1, C2);
}

Constant* LLVMContext::getConstantExprLShr(Constant* C1, Constant* C2) {
  return ConstantExpr::getLShr(C1, C2);
}

Constant* LLVMContext::getConstantExprAShr(Constant* C1, Constant* C2) {
  return ConstantExpr::getAShr(C1, C2);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Constant* const* IdxList, 
                                                    unsigned NumIdx) {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Value* const* IdxList, 
                                                    unsigned NumIdx) {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprExtractElement(Constant* Vec,
                                                     Constant* Idx) {
  return ConstantExpr::getExtractElement(Vec, Idx);
}

Constant* LLVMContext::getConstantExprInsertElement(Constant* Vec,
                                                    Constant* Elt,
                                                    Constant* Idx) {
  return ConstantExpr::getInsertElement(Vec, Elt, Idx);
}

Constant* LLVMContext::getConstantExprShuffleVector(Constant* V1, Constant* V2,
                                                    Constant* Mask) {
  return ConstantExpr::getShuffleVector(V1, V2, Mask);
}

Constant* LLVMContext::getConstantExprExtractValue(Constant* Agg,
                                                   const unsigned* IdxList, 
                                                   unsigned NumIdx) {
  return ConstantExpr::getExtractValue(Agg, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprInsertValue(Constant* Agg, Constant* Val,
                                                  const unsigned* IdxList,
                                                  unsigned NumIdx) {
  return ConstantExpr::getInsertValue(Agg, Val, IdxList, NumIdx);
}

Constant* LLVMContext::getZeroValueForNegation(const Type* Ty) {
  return ConstantExpr::getZeroValueForNegationExpr(Ty);
}


// ConstantFP accessors.
ConstantFP* LLVMContext::getConstantFP(const APFloat& V) {
  return ConstantFP::get(V);
}

Constant* LLVMContext::getConstantFP(const Type* Ty, double V) {
  return ConstantFP::get(Ty, V);
}

ConstantFP* LLVMContext::getConstantFPNegativeZero(const Type* Ty) {
  return ConstantFP::getNegativeZero(Ty);
}


// ConstantVector accessors.
Constant* LLVMContext::getConstantVector(const VectorType* T,
                            const std::vector<Constant*>& V) {
  return ConstantVector::get(T, V);
}

Constant* LLVMContext::getConstantVector(const std::vector<Constant*>& V) {
  return ConstantVector::get(V);
}

Constant* LLVMContext::getConstantVector(Constant* const* Vals,
                                         unsigned NumVals) {
  return ConstantVector::get(Vals, NumVals);
}

ConstantVector* LLVMContext::getConstantVectorAllOnes(const VectorType* Ty) {
  return ConstantVector::getAllOnesValue(Ty);
}

// MDNode accessors
MDNode* LLVMContext::getMDNode(Value* const* Vals, unsigned NumVals) {
  return MDNode::get(Vals, NumVals);
}

// MDString accessors
MDString* LLVMContext::getMDString(const char *StrBegin, const char *StrEnd) {
  return MDString::get(StrBegin, StrEnd);
}

MDString* LLVMContext::getMDString(const std::string &Str) {
  return MDString::get(Str);
}

// FunctionType accessors
FunctionType* LLVMContext::getFunctionType(const Type* Result,
                                         const std::vector<const Type*>& Params,
                                         bool isVarArg) {
  return FunctionType::get(Result, Params, isVarArg);
}
                                
// IntegerType accessors
const IntegerType* LLVMContext::getIntegerType(unsigned NumBits) {
  return IntegerType::get(NumBits);
}
  
// OpaqueType accessors
OpaqueType* LLVMContext::getOpaqueType() {
  return OpaqueType::get();
}

// StructType accessors
StructType* LLVMContext::getStructType(bool isPacked) {
  return StructType::get(isPacked);
}

StructType* LLVMContext::getStructType(const std::vector<const Type*>& Params,
                                       bool isPacked) {
  return StructType::get(Params, isPacked);
}

// ArrayType accessors
ArrayType* LLVMContext::getArrayType(const Type* ElementType,
                                     uint64_t NumElements) {
  return ArrayType::get(ElementType, NumElements);
}
  
// PointerType accessors
PointerType* LLVMContext::getPointerType(const Type* ElementType,
                                         unsigned AddressSpace) {
  return PointerType::get(ElementType, AddressSpace);
}

PointerType* LLVMContext::getPointerTypeUnqual(const Type* ElementType) {
  return PointerType::getUnqual(ElementType);
}
  
// VectorType accessors
VectorType* LLVMContext::getVectorType(const Type* ElementType,
                                       unsigned NumElements) {
  return VectorType::get(ElementType, NumElements);
}

VectorType* LLVMContext::getVectorTypeInteger(const VectorType* VTy) {
  return VectorType::getInteger(VTy);  
}

VectorType* LLVMContext::getVectorTypeExtendedElement(const VectorType* VTy) {
  return VectorType::getExtendedElementVectorType(VTy);
}

VectorType* LLVMContext::getVectorTypeTruncatedElement(const VectorType* VTy) {
  return VectorType::getTruncatedElementVectorType(VTy);
}
