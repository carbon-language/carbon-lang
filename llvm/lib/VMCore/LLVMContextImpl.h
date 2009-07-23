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

#include "llvm/LLVMContext.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/RWMutex.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <vector>

template<class ValType, class TypeClass, class ConstantClass,
         bool HasLargeKey = false  /*true for arrays and structs*/ >
class ValueMap;

namespace llvm {
template<class ValType>
struct ConstantTraits;

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
  sys::SmartRWMutex<true> ConstantsLock;
  
  typedef DenseMap<DenseMapAPIntKeyInfo::KeyTy, ConstantInt*, 
                   DenseMapAPIntKeyInfo> IntMapTy;
  IntMapTy IntConstants;
  
  typedef DenseMap<DenseMapAPFloatKeyInfo::KeyTy, ConstantFP*, 
                   DenseMapAPFloatKeyInfo> FPMapTy;
  FPMapTy FPConstants;
  
  StringMap<MDString*> MDStringCache;
  
  FoldingSet<MDNode> MDNodeSet;
  
  ValueMap<char, Type, ConstantAggregateZero> *AggZeroConstants;
  
  typedef ValueMap<std::vector<Constant*>, ArrayType, 
    ConstantArray, true /*largekey*/> ArrayConstantsTy;
  ArrayConstantsTy *ArrayConstants;
  
  typedef ValueMap<std::vector<Constant*>, StructType,
                   ConstantStruct, true /*largekey*/> StructConstantsTy;
  StructConstantsTy *StructConstants;
  
  LLVMContext &Context;
  ConstantInt *TheTrueVal;
  ConstantInt *TheFalseVal;
  
  LLVMContextImpl();
  LLVMContextImpl(const LLVMContextImpl&);
public:
  LLVMContextImpl(LLVMContext &C);
  ~LLVMContextImpl();
  
  /// Return a ConstantInt with the specified value and an implied Type. The
  /// type is the integer type that corresponds to the bit width of the value.
  ConstantInt *getConstantInt(const APInt &V);
  
  ConstantFP *getConstantFP(const APFloat &V);
  
  MDString *getMDString(const char *StrBegin, unsigned StrLength);
  
  MDNode *getMDNode(Value*const* Vals, unsigned NumVals);
  
  ConstantAggregateZero *getConstantAggregateZero(const Type *Ty);
  
  Constant *getConstantArray(const ArrayType *Ty,
                             const std::vector<Constant*> &V);
  
  Constant *getConstantStruct(const StructType *Ty, 
                              const std::vector<Constant*> &V);
  
  ConstantInt *getTrue() {
    if (TheTrueVal)
      return TheTrueVal;
    else
      return (TheTrueVal = Context.getConstantInt(IntegerType::get(1), 1));
  }
  
  ConstantInt *getFalse() {
    if (TheFalseVal)
      return TheFalseVal;
    else
      return (TheFalseVal = Context.getConstantInt(IntegerType::get(1), 0));
  }
  
  void erase(MDString *M);
  void erase(MDNode *M);
  void erase(ConstantAggregateZero *Z);
  void erase(ConstantArray *C);
  void erase(ConstantStruct *S);
  
  // RAUW helpers
  
  Constant *replaceUsesOfWithOnConstant(ConstantArray *CA, Value *From,
                                             Value *To, Use *U);
  Constant *replaceUsesOfWithOnConstant(ConstantStruct *CS, Value *From,
                                        Value *To, Use *U);
};

}

#endif
