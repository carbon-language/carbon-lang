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
#include "llvm/System/Mutex.h"
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
  
  // Lock used for guarding access to the type maps.
  sys::SmartMutex<true> TypeMapLock;
  
  TypeMap<ArrayValType, ArrayType> ArrayTypes;
  TypeMap<VectorValType, VectorType> VectorTypes;
  TypeMap<PointerValType, PointerType> PointerTypes;
  TypeMap<FunctionValType, FunctionType> FunctionTypes;
  TypeMap<StructValType, StructType> StructTypes;
  TypeMap<IntegerValType, IntegerType> IntegerTypes;
  
  const Type *VoidTy;
  const Type *LabelTy;
  const Type *FloatTy;
  const Type *DoubleTy;
  const Type *MetadataTy;
  const Type *X86_FP80Ty;
  const Type *FP128Ty;
  const Type *PPC_FP128Ty;
  
  const IntegerType *Int1Ty;
  const IntegerType *Int8Ty;
  const IntegerType *Int16Ty;
  const IntegerType *Int32Ty;
  const IntegerType *Int64Ty;
  
  LLVMContextImpl(LLVMContext &C) : TheTrueVal(0), TheFalseVal(0),
    VoidTy(new Type(C, Type::VoidTyID)),
    LabelTy(new Type(C, Type::LabelTyID)),
    FloatTy(new Type(C, Type::FloatTyID)),
    DoubleTy(new Type(C, Type::DoubleTyID)),
    MetadataTy(new Type(C, Type::MetadataTyID)),
    X86_FP80Ty(new Type(C, Type::X86_FP80TyID)),
    FP128Ty(new Type(C, Type::FP128TyID)),
    PPC_FP128Ty(new Type(C, Type::PPC_FP128TyID)),
    Int1Ty(new IntegerType(C, 1)),
    Int8Ty(new IntegerType(C, 8)),
    Int16Ty(new IntegerType(C, 16)),
    Int32Ty(new IntegerType(C, 32)),
    Int64Ty(new IntegerType(C, 64)) { }
  
  ~LLVMContextImpl() {
    // In principle, we should delete the member types here.  However,
    // this causes destruction order issues with the types in the TypeMaps.
    // For now, just leak this, which is at least not a regression from the
    // previous behavior, though still undesirable.
#if 0
    delete VoidTy;
    delete LabelTy;
    delete FloatTy;
    delete DoubleTy;
    delete MetadataTy;
    delete X86_FP80Ty;
    delete FP128Ty;
    delete PPC_FP128Ty;
    
    delete Int1Ty;
    delete Int8Ty;
    delete Int16Ty;
    delete Int32Ty;
    delete Int64Ty;
#endif
  }
};

}

#endif
