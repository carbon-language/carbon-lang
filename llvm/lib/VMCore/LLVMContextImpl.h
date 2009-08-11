//===----------------- LLVMContextImpl.h - Implementation ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_IMPL_H
#define LLVM_LLVMCONTEXT_IMPL_H

#include "ConstantsContext.h"
#include "TypesContext.h"
#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/System/RWMutex.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringMap.h"
#include <vector>

namespace llvm {

class ConstantInt;
class ConstantFP;
class MDString;
class MDNode;
class LLVMContext;
class Type;
class Value;

struct DenseMapAPIntKeyInfo {
  struct KeyTy {
    APInt val;
    const Type* type;
    KeyTy(const APInt& V, const Type* Ty) : val(V), type(Ty) {}
    KeyTy(const KeyTy& that) : val(that.val), type(that.type) {}
    bool operator==(const KeyTy& that) const {
      return type == that.type && this->val == that.val;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline KeyTy getEmptyKey() { return KeyTy(APInt(1,0), 0); }
  static inline KeyTy getTombstoneKey() { return KeyTy(APInt(1,1), 0); }
  static unsigned getHashValue(const KeyTy &Key) {
    return DenseMapInfo<void*>::getHashValue(Key.type) ^ 
      Key.val.getHashValue();
  }
  static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return false; }
};

struct DenseMapAPFloatKeyInfo {
  struct KeyTy {
    APFloat val;
    KeyTy(const APFloat& V) : val(V){}
    KeyTy(const KeyTy& that) : val(that.val) {}
    bool operator==(const KeyTy& that) const {
      return this->val.bitwiseIsEqual(that.val);
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline KeyTy getEmptyKey() { 
    return KeyTy(APFloat(APFloat::Bogus,1));
  }
  static inline KeyTy getTombstoneKey() { 
    return KeyTy(APFloat(APFloat::Bogus,2)); 
  }
  static unsigned getHashValue(const KeyTy &Key) {
    return Key.val.getHashValue();
  }
  static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return false; }
};

class LLVMContextImpl {
public:
  sys::SmartRWMutex<true> ConstantsLock;
  
  typedef DenseMap<DenseMapAPIntKeyInfo::KeyTy, ConstantInt*, 
                         DenseMapAPIntKeyInfo> IntMapTy;
  IntMapTy IntConstants;
  
  typedef DenseMap<DenseMapAPFloatKeyInfo::KeyTy, ConstantFP*, 
                         DenseMapAPFloatKeyInfo> FPMapTy;
  FPMapTy FPConstants;
  
  StringMap<MDString*> MDStringCache;
  
  ValueMap<char, Type, ConstantAggregateZero> AggZeroConstants;

  typedef ValueMap<std::vector<Value*>, Type, MDNode, true /*largekey*/> 
  MDNodeMapTy;

  MDNodeMapTy MDNodes;
  
  typedef ValueMap<std::vector<Constant*>, ArrayType, 
    ConstantArray, true /*largekey*/> ArrayConstantsTy;
  ArrayConstantsTy ArrayConstants;
  
  typedef ValueMap<std::vector<Constant*>, StructType,
                   ConstantStruct, true /*largekey*/> StructConstantsTy;
  StructConstantsTy StructConstants;
  
  typedef ValueMap<std::vector<Constant*>, VectorType,
                   ConstantVector> VectorConstantsTy;
  VectorConstantsTy VectorConstants;
  
  ValueMap<char, PointerType, ConstantPointerNull> NullPtrConstants;
  
  ValueMap<char, Type, UndefValue> UndefValueConstants;
  
  ValueMap<ExprMapKeyType, Type, ConstantExpr> ExprConstants;
  
  ConstantInt *TheTrueVal;
  ConstantInt *TheFalseVal;
  
  TypeMap<ArrayValType, ArrayType> ArrayTypes;
  TypeMap<VectorValType, VectorType> VectorTypes;
  TypeMap<PointerValType, PointerType> PointerTypes;
  TypeMap<FunctionValType, FunctionType> FunctionTypes;
  TypeMap<StructValType, StructType> StructTypes;
  
  LLVMContextImpl() : TheTrueVal(0), TheFalseVal(0) { }
};

}

#endif
