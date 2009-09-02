//===-- LLVMContextImpl.h - The LLVMContextImpl opaque class --------------===//
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
#include "LeaksContext.h"
#include "TypesContext.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/RWMutex.h"
#include "llvm/Assembly/Writer.h"
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
  
  FoldingSet<MDNode> MDNodeSet;
  
  ValueMap<char, Type, ConstantAggregateZero> AggZeroConstants;

  SmallPtrSet<const MDNode *, 8> MDNodes;

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
  
  // Lock used for guarding access to the leak detector
  sys::SmartMutex<true> LLVMObjectsLock;
  LeakDetectorImpl<Value> LLVMObjects;
  
  // Lock used for guarding access to the type maps.
  sys::SmartMutex<true> TypeMapLock;
  
  // Recursive lock used for guarding access to AbstractTypeUsers.
  // NOTE: The true template parameter means this will no-op when we're not in
  // multithreaded mode.
  sys::SmartMutex<true> AbstractTypeUsersLock;

  // Basic type instances.
  const Type VoidTy;
  const Type LabelTy;
  const Type FloatTy;
  const Type DoubleTy;
  const Type MetadataTy;
  const Type X86_FP80Ty;
  const Type FP128Ty;
  const Type PPC_FP128Ty;
  const IntegerType Int1Ty;
  const IntegerType Int8Ty;
  const IntegerType Int16Ty;
  const IntegerType Int32Ty;
  const IntegerType Int64Ty;

  // Concrete/Abstract TypeDescriptions - We lazily calculate type descriptions
  // for types as they are needed.  Because resolution of types must invalidate
  // all of the abstract type descriptions, we keep them in a seperate map to 
  // make this easy.
  TypePrinting ConcreteTypeDescriptions;
  TypePrinting AbstractTypeDescriptions;
  
  TypeMap<ArrayValType, ArrayType> ArrayTypes;
  TypeMap<VectorValType, VectorType> VectorTypes;
  TypeMap<PointerValType, PointerType> PointerTypes;
  TypeMap<FunctionValType, FunctionType> FunctionTypes;
  TypeMap<StructValType, StructType> StructTypes;
  TypeMap<IntegerValType, IntegerType> IntegerTypes;

  /// ValueHandles - This map keeps track of all of the value handles that are
  /// watching a Value*.  The Value::HasValueHandle bit is used to know
  // whether or not a value has an entry in this map.
  typedef DenseMap<Value*, ValueHandleBase*> ValueHandlesTy;
  ValueHandlesTy ValueHandles;
  
  LLVMContextImpl(LLVMContext &C) : TheTrueVal(0), TheFalseVal(0),
    VoidTy(C, Type::VoidTyID),
    LabelTy(C, Type::LabelTyID),
    FloatTy(C, Type::FloatTyID),
    DoubleTy(C, Type::DoubleTyID),
    MetadataTy(C, Type::MetadataTyID),
    X86_FP80Ty(C, Type::X86_FP80TyID),
    FP128Ty(C, Type::FP128TyID),
    PPC_FP128Ty(C, Type::PPC_FP128TyID),
    Int1Ty(C, 1),
    Int8Ty(C, 8),
    Int16Ty(C, 16),
    Int32Ty(C, 32),
    Int64Ty(C, 64) { }

  ~LLVMContextImpl()
  {
    ExprConstants.freeConstants();
    ArrayConstants.freeConstants();
    StructConstants.freeConstants();
    VectorConstants.freeConstants();
    AggZeroConstants.freeConstants();
    NullPtrConstants.freeConstants();
    UndefValueConstants.freeConstants();
    for (IntMapTy::iterator I=IntConstants.begin(), E=IntConstants.end(); 
         I != E; ++I) {
      if (I->second->use_empty())
        delete I->second;
    }
    for (FPMapTy::iterator I=FPConstants.begin(), E=FPConstants.end(); 
         I != E; ++I) {
      if (I->second->use_empty())
        delete I->second;
    }
  }
};

}

#endif
