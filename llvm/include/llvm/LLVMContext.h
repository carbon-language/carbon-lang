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
  
  /// @returns the value for an integer constant of the given type that has all
  /// its bits set to true.
  /// @brief Get the all ones value
  Constant* getAllOnesValue(const Type* Ty);
  
  // UndefValue accessors
  UndefValue* getUndef(const Type* Ty);
  
  // ConstantInt accessors
  ConstantInt* getConstantIntTrue();
  ConstantInt* getConstantIntFalse();
  
  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  Constant* getConstantInt(const Type* Ty, uint64_t V,
                              bool isSigned = false);
                              
  /// Return a ConstantInt with the specified integer value for the specified
  /// type. If the type is wider than 64 bits, the value will be zero-extended
  /// to fit the type, unless isSigned is true, in which case the value will
  /// be interpreted as a 64-bit signed integer and sign-extended to fit
  /// the type.
  /// @brief Get a ConstantInt for a specific value.
  ConstantInt* getConstantInt(const IntegerType* Ty, uint64_t V,
                              bool isSigned = false);

  /// Return a ConstantInt with the specified value for the specified type. The
  /// value V will be canonicalized to a an unsigned APInt. Accessing it with
  /// either getSExtValue() or getZExtValue() will yield a correctly sized and
  /// signed value for the type Ty.
  /// @brief Get a ConstantInt for a specific signed value.
  ConstantInt* getConstantIntSigned(const IntegerType* Ty, int64_t V);
  Constant *getConstantIntSigned(const Type *Ty, int64_t V);
  
  ConstantInt* getConstantInt(const APInt& V);
  
  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  Constant* getConstantInt(const Type* Ty, const APInt& V);
  
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
                             
  /// This method constructs a ConstantArray and initializes it with a text
  /// string. The default behavior (AddNull==true) causes a null terminator to
  /// be placed at the end of the array. This effectively increases the length
  /// of the array by one (you've been warned).  However, in some situations 
  /// this is not desired so if AddNull==false then the string is copied without
  /// null termination.
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
  
  /// getAlignOf constant expr - computes the alignment of a type in a target
  /// independent way (Note: the return type is an i32; Note: assumes that i8
  /// is byte aligned).
  ///
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

  /// getSizeOf constant expr - computes the size of a type in a target
  /// independent way (Note: the return type is an i64).
  ///
  Constant* getConstantExprSizeOf(const Type* Ty);
  
  /// Floating point negation must be implemented with f(x) = -0.0 - x. This
  /// method returns the negative zero constant for floating point or vector
  /// floating point types; for all other types, it returns the null value.
  Constant* getZeroValueForNegation(const Type* Ty);
  
  // ConstantFP accessors
  ConstantFP* getConstantFP(const APFloat& V);
  
  /// get() - This returns a ConstantFP, or a vector containing a splat of a
  /// ConstantFP, for the specified value in the specified type.  This should
  /// only be used for simple constant values like 2.0/1.0 etc, that are
  /// known-valid both as host double and as the target format.
  Constant* getConstantFP(const Type* Ty, double V);
  ConstantFP* getConstantFPNegativeZero(const Type* Ty);
  
  // ConstantVector accessors
  Constant* getConstantVector(const VectorType* T,
                              const std::vector<Constant*>& V);
  Constant* getConstantVector(const std::vector<Constant*>& V);
  Constant* getConstantVector(Constant* const* Vals, unsigned NumVals);
  
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
  StructType* getStructType(const Type* type, ...);
  
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
  
  // Other helpers
  /// @brief Create a result type for fcmp/icmp
  const Type* makeCmpResultType(const Type* opnd_type);
};

/// FOR BACKWARDS COMPATIBILITY - Returns a global context.
extern LLVMContext& getGlobalContext();

}

#endif
