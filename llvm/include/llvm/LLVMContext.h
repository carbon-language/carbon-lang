//===-- llvm/LLVMContext.h - Class for managing "global" state --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_H
#define LLVM_LLVMCONTEXT_H

#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class LLVMContextImpl;
class Constant;
class ConstantInt;
class ConstantPointerNull;
class ConstantStruct;
class ConstantAggregateZero;
class ConstantArray;
class ConstantFP;
class ConstantVector;
class IntegerType;
class PointerType;
class StructType;
class ArrayType;
class VectorType;
class OpaqueType;
class FunctionType;
class Type;
class APInt;
class APFloat;
class Value;

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core 
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
  LLVMContextImpl* pImpl;
public:
  LLVMContext();
  ~LLVMContext();
  
  // ConstantInt accessors
  ConstantInt* getConstantIntTrue() const;
  ConstantInt* getConstantIntFalse() const;
  ConstantInt* getConstantInt(const IntegerType* Ty, uint64_t V,
                              bool isSigned = false) const;
  ConstantInt* getConstantIntSigned(const IntegerType* Ty, int64_t V) const;
  ConstantInt* getConstantInt(const APInt& V) const;
  Constant* getConstantInt(const Type* Ty, const APInt& V) const;
  ConstantInt* getAllOnesConstantInt(const Type* Ty) const;
  
  // ConstantPointerNull accessors
  ConstantPointerNull* getConstantPointerNull(const PointerType* T) const;
  
  // ConstantStruct accessors
  Constant* getConstantStruct(const StructType* T,
                              const std::vector<Constant*>& V) const;
  Constant* getConstantStruct(const std::vector<Constant*>& V,
                              bool Packed = false) const;
  Constant* getConstantStruct(Constant* const *Vals, unsigned NumVals,
                              bool Packed = false) const;
                              
  // ConstantAggregateZero accessors
  ConstantAggregateZero* getConstantAggregateZero(const Type* Ty) const;
  
  // ConstantArray accessors
  Constant* getConstantArray(const ArrayType* T,
                             const std::vector<Constant*>& V) const;
  Constant* getConstantArray(const ArrayType* T, Constant* const* Vals,
                             unsigned NumVals) const;
  Constant* getConstantArray(const std::string& Initializer,
                             bool AddNull = false) const;
                             
  // ConstantExpr accessors
  Constant* getConstantExpr(unsigned Opcode, Constant* C1, Constant* C2) const;
  Constant* getConstantExprTrunc(Constant* C, const Type* Ty) const;
  Constant* getConstantExprSExt(Constant* C, const Type* Ty) const;
  Constant* getConstantExprZExt(Constant* C, const Type* Ty) const;
  Constant* getConstantExprFPTrunc(Constant* C, const Type* Ty) const;
  Constant* getConstantExprFPExtend(Constant* C, const Type* Ty) const;
  Constant* getConstantExprUIToFP(Constant* C, const Type* Ty) const;
  Constant* getConstantExprSIToFP(Constant* C, const Type* Ty) const;
  Constant* getConstantExprFPToUI(Constant* C, const Type* Ty) const;
  Constant* getConstantExprFPToSI(Constant* C, const Type* Ty) const;
  Constant* getConstantExprPtrToInt(Constant* C, const Type* Ty) const;
  Constant* getConstantExprIntToPtr(Constant* C, const Type* Ty) const;
  Constant* getConstantExprBitCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprCast(unsigned ops, Constant* C,
                                const Type* Ty) const;
  Constant* getConstantExprZExtOrBitCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprSExtOrBitCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprTruncOrBitCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprPointerCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprIntegerCast(Constant* C, const Type* Ty,
                                       bool isSigned) const;
  Constant* getConstantExprFPCast(Constant* C, const Type* Ty) const;
  Constant* getConstantExprSelect(Constant* C, Constant* V1,
                                  Constant* V2) const;
  Constant* getConstantExprAlignOf(const Type* Ty) const;
  Constant* getConstantExprCompare(unsigned short pred,
                                   Constant* C1, Constant* C2) const;
  Constant* getConstantExprNeg(Constant* C) const;
  Constant* getConstantExprFNeg(Constant* C) const;
  Constant* getConstantExprNot(Constant* C) const;
  Constant* getConstantExprAdd(Constant* C1, Constant* C2) const;
  Constant* getConstantExprFAdd(Constant* C1, Constant* C2) const;
  Constant* getConstantExprSub(Constant* C1, Constant* C2) const;
  Constant* getConstantExprFSub(Constant* C1, Constant* C2) const;
  Constant* getConstantExprMul(Constant* C1, Constant* C2) const;
  Constant* getConstantExprFMul(Constant* C1, Constant* C2) const;
  Constant* getConstantExprUDiv(Constant* C1, Constant* C2) const;
  Constant* getConstantExprSDiv(Constant* C1, Constant* C2) const;
  Constant* getConstantExprFDiv(Constant* C1, Constant* C2) const;
  Constant* getConstantExprURem(Constant* C1, Constant* C2) const;
  Constant* getConstantExprSRem(Constant* C1, Constant* C2) const;
  Constant* getConstantExprFRem(Constant* C1, Constant* C2) const;
  Constant* getConstantExprAnd(Constant* C1, Constant* C2) const;
  Constant* getConstantExprOr(Constant* C1, Constant* C2) const;
  Constant* getConstantExprXor(Constant* C1, Constant* C2) const;
  Constant* getConstantExprICmp(unsigned short pred, Constant* LHS,
                                Constant* RHS) const;
  Constant* getConstantExprFCmp(unsigned short pred, Constant* LHS,
                                Constant* RHS) const;
  Constant* getConstantExprVICmp(unsigned short pred, Constant* LHS,
                                 Constant* RHS) const;
  Constant* getConstantExprVFCmp(unsigned short pred, Constant* LHS,
                                 Constant* RHS) const;
  Constant* getConstantExprShl(Constant* C1, Constant* C2) const;
  Constant* getConstantExprLShr(Constant* C1, Constant* C2) const;
  Constant* getConstantExprAShr(Constant* C1, Constant* C2) const;
  Constant* getConstantExprGetElementPtr(Constant* C, Constant* const* IdxList, 
                                         unsigned NumIdx) const;
  Constant* getConstantExprGetElementPtr(Constant* C, Value* const* IdxList, 
                                          unsigned NumIdx) const;
  Constant* getConstantExprExtractElement(Constant* Vec, Constant* Idx) const;
  Constant* getConstantExprInsertElement(Constant* Vec, Constant* Elt,
                                         Constant* Idx) const;
  Constant* getConstantExprShuffleVector(Constant* V1, Constant* V2,
                                         Constant* Mask) const;
  Constant* getConstantExprExtractValue(Constant* Agg, const unsigned* IdxList, 
                                        unsigned NumIdx) const;
  Constant* getConstantExprInsertValue(Constant* Agg, Constant* Val,
                                       const unsigned* IdxList,
                                       unsigned NumIdx) const;
  Constant* getZeroValueForNegation(const Type* Ty) const;
  
  // ConstantFP accessors
  ConstantFP* getConstantFP(const APFloat& V) const;
  Constant* getConstantFP(const Type* Ty, double V) const;
  ConstantFP* getConstantFPNegativeZero(const Type* Ty) const;
  
  // ConstantVector accessors
  Constant* getConstantVector(const VectorType* T,
                              const std::vector<Constant*>& V) const;
  Constant* getConstantVector(const std::vector<Constant*>& V) const;
  Constant* getConstantVector(Constant* const* Vals, unsigned NumVals) const;
  ConstantVector* getConstantVectorAllOnes(const VectorType* Ty) const;
  
  // FunctionType accessors
  FunctionType* getFunctionType(const Type* Result,
                                const std::vector<const Type*>& Params,
                                bool isVarArg) const;
                                
  // IntegerType accessors
  const IntegerType* getIntegerType(unsigned NumBits) const;
  
  // OpaqueType accessors
  OpaqueType* getOpaqueType() const;
  
  // StructType accessors
  StructType* getStructType(const std::vector<const Type*>& Params,
                            bool isPacked = false) const;
  
  // ArrayType accessors
  ArrayType* getArrayType(const Type* ElementType, uint64_t NumElements) const;
  
  // PointerType accessors
  PointerType* getPointerType(const Type* ElementType,
                              unsigned AddressSpace) const;
  PointerType* getPointerTypeUnqualified(const Type* ElementType) const;
  
  // VectorType accessors
  VectorType* getVectorType(const Type* ElementType,
                            unsigned NumElements) const;
  VectorType* getVectorTypeInteger(const VectorType* VTy) const;
  VectorType* getVectorTypeExtendedElement(const VectorType* VTy) const;
  VectorType* getVectorTypeTruncatedElement(const VectorType* VTy) const;
};

/// FOR BACKWARDS COMPATIBILITY - Returns a global context.
extern const LLVMContext& getGlobalContext();

}

#endif
