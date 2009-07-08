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
class UndefValue;
class MDNode;
class MDString;
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
  
  // Constant accessors
  Constant* getNullValue(const Type* Ty);
  Constant* getAllOnesValue(const Type* Ty);
  
  // UndefValue accessors
  UndefValue* getUndef(const Type* Ty);
  
  // ConstantInt accessors
  ConstantInt* getConstantIntTrue();
  ConstantInt* getConstantIntFalse();
  Constant* getConstantInt(const Type* Ty, uint64_t V,
                              bool isSigned = false);
  ConstantInt* getConstantInt(const IntegerType* Ty, uint64_t V,
                              bool isSigned = false);
  ConstantInt* getConstantIntSigned(const IntegerType* Ty, int64_t V);
  ConstantInt* getConstantInt(const APInt& V);
  Constant* getConstantInt(const Type* Ty, const APInt& V);
  ConstantInt* getConstantIntAllOnesValue(const Type* Ty);
  
  // ConstantPointerNull accessors
  ConstantPointerNull* getConstantPointerNull(const PointerType* T);
  
  // ConstantStruct accessors
  Constant* getConstantStruct(const StructType* T,
                              const std::vector<Constant*>& V);
  Constant* getConstantStruct(const std::vector<Constant*>& V,
                              bool Packed = false);
  Constant* getConstantStruct(Constant* const *Vals, unsigned NumVals,
                              bool Packed = false);
                              
  // ConstantAggregateZero accessors
  ConstantAggregateZero* getConstantAggregateZero(const Type* Ty);
  
  // ConstantArray accessors
  Constant* getConstantArray(const ArrayType* T,
                             const std::vector<Constant*>& V);
  Constant* getConstantArray(const ArrayType* T, Constant* const* Vals,
                             unsigned NumVals);
  Constant* getConstantArray(const std::string& Initializer,
                             bool AddNull = true);
                             
  // ConstantExpr accessors
  Constant* getConstantExpr(unsigned Opcode, Constant* C1, Constant* C2);
  Constant* getConstantExprTrunc(Constant* C, const Type* Ty);
  Constant* getConstantExprSExt(Constant* C, const Type* Ty);
  Constant* getConstantExprZExt(Constant* C, const Type* Ty);
  Constant* getConstantExprFPTrunc(Constant* C, const Type* Ty);
  Constant* getConstantExprFPExtend(Constant* C, const Type* Ty);
  Constant* getConstantExprUIToFP(Constant* C, const Type* Ty);
  Constant* getConstantExprSIToFP(Constant* C, const Type* Ty);
  Constant* getConstantExprFPToUI(Constant* C, const Type* Ty);
  Constant* getConstantExprFPToSI(Constant* C, const Type* Ty);
  Constant* getConstantExprPtrToInt(Constant* C, const Type* Ty);
  Constant* getConstantExprIntToPtr(Constant* C, const Type* Ty);
  Constant* getConstantExprBitCast(Constant* C, const Type* Ty);
  Constant* getConstantExprCast(unsigned ops, Constant* C, const Type* Ty);
  Constant* getConstantExprZExtOrBitCast(Constant* C, const Type* Ty);
  Constant* getConstantExprSExtOrBitCast(Constant* C, const Type* Ty);
  Constant* getConstantExprTruncOrBitCast(Constant* C, const Type* Ty);
  Constant* getConstantExprPointerCast(Constant* C, const Type* Ty);
  Constant* getConstantExprIntegerCast(Constant* C, const Type* Ty,
                                       bool isSigned);
  Constant* getConstantExprFPCast(Constant* C, const Type* Ty);
  Constant* getConstantExprSelect(Constant* C, Constant* V1, Constant* V2);
  Constant* getConstantExprAlignOf(const Type* Ty);
  Constant* getConstantExprCompare(unsigned short pred,
                                   Constant* C1, Constant* C2);
  Constant* getConstantExprNeg(Constant* C);
  Constant* getConstantExprFNeg(Constant* C);
  Constant* getConstantExprNot(Constant* C);
  Constant* getConstantExprAdd(Constant* C1, Constant* C2);
  Constant* getConstantExprFAdd(Constant* C1, Constant* C2);
  Constant* getConstantExprSub(Constant* C1, Constant* C2);
  Constant* getConstantExprFSub(Constant* C1, Constant* C2);
  Constant* getConstantExprMul(Constant* C1, Constant* C2);
  Constant* getConstantExprFMul(Constant* C1, Constant* C2);
  Constant* getConstantExprUDiv(Constant* C1, Constant* C2);
  Constant* getConstantExprSDiv(Constant* C1, Constant* C2);
  Constant* getConstantExprFDiv(Constant* C1, Constant* C2);
  Constant* getConstantExprURem(Constant* C1, Constant* C2);
  Constant* getConstantExprSRem(Constant* C1, Constant* C2);
  Constant* getConstantExprFRem(Constant* C1, Constant* C2);
  Constant* getConstantExprAnd(Constant* C1, Constant* C2);
  Constant* getConstantExprOr(Constant* C1, Constant* C2);
  Constant* getConstantExprXor(Constant* C1, Constant* C2);
  Constant* getConstantExprICmp(unsigned short pred, Constant* LHS,
                                Constant* RHS);
  Constant* getConstantExprFCmp(unsigned short pred, Constant* LHS,
                                Constant* RHS);
  Constant* getConstantExprShl(Constant* C1, Constant* C2);
  Constant* getConstantExprLShr(Constant* C1, Constant* C2);
  Constant* getConstantExprAShr(Constant* C1, Constant* C2);
  Constant* getConstantExprGetElementPtr(Constant* C, Constant* const* IdxList, 
                                         unsigned NumIdx);
  Constant* getConstantExprGetElementPtr(Constant* C, Value* const* IdxList, 
                                          unsigned NumIdx);
  Constant* getConstantExprExtractElement(Constant* Vec, Constant* Idx);
  Constant* getConstantExprInsertElement(Constant* Vec, Constant* Elt,
                                         Constant* Idx);
  Constant* getConstantExprShuffleVector(Constant* V1, Constant* V2,
                                         Constant* Mask);
  Constant* getConstantExprExtractValue(Constant* Agg, const unsigned* IdxList, 
                                        unsigned NumIdx);
  Constant* getConstantExprInsertValue(Constant* Agg, Constant* Val,
                                       const unsigned* IdxList,
                                       unsigned NumIdx);
  Constant* getConstantExprSizeOf(const Type* Ty);
  Constant* getZeroValueForNegation(const Type* Ty);
  
  // ConstantFP accessors
  ConstantFP* getConstantFP(const APFloat& V);
  Constant* getConstantFP(const Type* Ty, double V);
  ConstantFP* getConstantFPNegativeZero(const Type* Ty);
  
  // ConstantVector accessors
  Constant* getConstantVector(const VectorType* T,
                              const std::vector<Constant*>& V);
  Constant* getConstantVector(const std::vector<Constant*>& V);
  Constant* getConstantVector(Constant* const* Vals, unsigned NumVals);
  ConstantVector* getConstantVectorAllOnesValue(const VectorType* Ty);
  
  // MDNode accessors
  MDNode* getMDNode(Value* const* Vals, unsigned NumVals);
  
  // MDString accessors
  MDString* getMDString(const char *StrBegin, const char *StrEnd);
  MDString* getMDString(const std::string &Str);
  
  // FunctionType accessors
  FunctionType* getFunctionType(const Type* Result, bool isVarArg);
  FunctionType* getFunctionType(const Type* Result,
                                const std::vector<const Type*>& Params,
                                bool isVarArg);
                                
  // IntegerType accessors
  const IntegerType* getIntegerType(unsigned NumBits);
  
  // OpaqueType accessors
  OpaqueType* getOpaqueType();
  
  // StructType accessors
  StructType* getStructType(bool isPacked=false);
  StructType* getStructType(const std::vector<const Type*>& Params,
                            bool isPacked = false);
  
  // ArrayType accessors
  ArrayType* getArrayType(const Type* ElementType, uint64_t NumElements);
  
  // PointerType accessors
  PointerType* getPointerType(const Type* ElementType, unsigned AddressSpace);
  PointerType* getPointerTypeUnqual(const Type* ElementType);
  
  // VectorType accessors
  VectorType* getVectorType(const Type* ElementType, unsigned NumElements);
  VectorType* getVectorTypeInteger(const VectorType* VTy);
  VectorType* getVectorTypeExtendedElement(const VectorType* VTy);
  VectorType* getVectorTypeTruncatedElement(const VectorType* VTy);
};

/// FOR BACKWARDS COMPATIBILITY - Returns a global context.
extern LLVMContext& getGlobalContext();

}

#endif
