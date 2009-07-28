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
#include "llvm/Constants.h"
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

namespace llvm {
template<class ValType>
struct ConstantTraits;

// The number of operands for each ConstantCreator::create method is
// determined by the ConstantTraits template.
// ConstantCreator - A class that is used to create constants by
// ValueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
template<typename T, typename Alloc>
struct VISIBILITY_HIDDEN ConstantTraits< std::vector<T, Alloc> > {
  static unsigned uses(const std::vector<T, Alloc>& v) {
    return v.size();
  }
};

template<class ConstantClass, class TypeClass, class ValType>
struct VISIBILITY_HIDDEN ConstantCreator {
  static ConstantClass *create(const TypeClass *Ty, const ValType &V) {
    return new(ConstantTraits<ValType>::uses(V)) ConstantClass(Ty, V);
  }
};

template<class ConstantClass, class TypeClass>
struct VISIBILITY_HIDDEN ConvertConstantType {
  static void convert(ConstantClass *OldC, const TypeClass *NewTy) {
    llvm_unreachable("This type cannot be converted!");
  }
};

// ConstantAggregateZero does not take extra "value" argument...
template<class ValType>
struct ConstantCreator<ConstantAggregateZero, Type, ValType> {
  static ConstantAggregateZero *create(const Type *Ty, const ValType &V){
    return new ConstantAggregateZero(Ty);
  }
};

template<>
struct ConvertConstantType<ConstantAggregateZero, Type> {
  static void convert(ConstantAggregateZero *OldC, const Type *NewTy) {
    // Make everyone now use a constant of the new type...
    Constant *New = NewTy->getContext().getConstantAggregateZero(NewTy);
    assert(New != OldC && "Didn't replace constant??");
    OldC->uncheckedReplaceAllUsesWith(New);
    OldC->destroyConstant();     // This constant is now dead, destroy it.
  }
};

template<>
struct ConvertConstantType<ConstantArray, ArrayType> {
  static void convert(ConstantArray *OldC, const ArrayType *NewTy) {
    // Make everyone now use a constant of the new type...
    std::vector<Constant*> C;
    for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
      C.push_back(cast<Constant>(OldC->getOperand(i)));
    Constant *New = ConstantArray::get(NewTy, C);
    assert(New != OldC && "Didn't replace constant??");
    OldC->uncheckedReplaceAllUsesWith(New);
    OldC->destroyConstant();    // This constant is now dead, destroy it.
  }
};

template<>
struct ConvertConstantType<ConstantStruct, StructType> {
  static void convert(ConstantStruct *OldC, const StructType *NewTy) {
    // Make everyone now use a constant of the new type...
    std::vector<Constant*> C;
    for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
      C.push_back(cast<Constant>(OldC->getOperand(i)));
    Constant *New = ConstantStruct::get(NewTy, C);
    assert(New != OldC && "Didn't replace constant??");

    OldC->uncheckedReplaceAllUsesWith(New);
    OldC->destroyConstant();    // This constant is now dead, destroy it.
  }
};

template<>
struct ConvertConstantType<ConstantVector, VectorType> {
  static void convert(ConstantVector *OldC, const VectorType *NewTy) {
    // Make everyone now use a constant of the new type...
    std::vector<Constant*> C;
    for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
      C.push_back(cast<Constant>(OldC->getOperand(i)));
    Constant *New = ConstantVector::get(NewTy, C);
    assert(New != OldC && "Didn't replace constant??");
    OldC->uncheckedReplaceAllUsesWith(New);
    OldC->destroyConstant();    // This constant is now dead, destroy it.
  }
};

template<class ValType, class TypeClass, class ConstantClass,
         bool HasLargeKey = false /*true for arrays and structs*/ >
class ValueMap : public AbstractTypeUser {
public:
  typedef std::pair<const Type*, ValType> MapKey;
  typedef std::map<MapKey, Constant *> MapTy;
  typedef std::map<Constant*, typename MapTy::iterator> InverseMapTy;
  typedef std::map<const Type*, typename MapTy::iterator> AbstractTypeMapTy;
private:
  /// Map - This is the main map from the element descriptor to the Constants.
  /// This is the primary way we avoid creating two of the same shape
  /// constant.
  MapTy Map;
    
  /// InverseMap - If "HasLargeKey" is true, this contains an inverse mapping
  /// from the constants to their element in Map.  This is important for
  /// removal of constants from the array, which would otherwise have to scan
  /// through the map with very large keys.
  InverseMapTy InverseMap;

  /// AbstractTypeMap - Map for abstract type constants.
  ///
  AbstractTypeMapTy AbstractTypeMap;
    
  /// ValueMapLock - Mutex for this map.
  sys::SmartMutex<true> ValueMapLock;

public:
  // NOTE: This function is not locked.  It is the caller's responsibility
  // to enforce proper synchronization.
  typename MapTy::iterator map_end() { return Map.end(); }
    
  /// InsertOrGetItem - Return an iterator for the specified element.
  /// If the element exists in the map, the returned iterator points to the
  /// entry and Exists=true.  If not, the iterator points to the newly
  /// inserted entry and returns Exists=false.  Newly inserted entries have
  /// I->second == 0, and should be filled in.
  /// NOTE: This function is not locked.  It is the caller's responsibility
  // to enforce proper synchronization.
  typename MapTy::iterator InsertOrGetItem(std::pair<MapKey, Constant *>
                                 &InsertVal,
                                 bool &Exists) {
    std::pair<typename MapTy::iterator, bool> IP = Map.insert(InsertVal);
    Exists = !IP.second;
    return IP.first;
  }
    
private:
  typename MapTy::iterator FindExistingElement(ConstantClass *CP) {
    if (HasLargeKey) {
      typename InverseMapTy::iterator IMI = InverseMap.find(CP);
      assert(IMI != InverseMap.end() && IMI->second != Map.end() &&
             IMI->second->second == CP &&
             "InverseMap corrupt!");
      return IMI->second;
    }
      
    typename MapTy::iterator I =
      Map.find(MapKey(static_cast<const TypeClass*>(CP->getRawType()),
                      getValType(CP)));
    if (I == Map.end() || I->second != CP) {
      // FIXME: This should not use a linear scan.  If this gets to be a
      // performance problem, someone should look at this.
      for (I = Map.begin(); I != Map.end() && I->second != CP; ++I)
        /* empty */;
    }
    return I;
  }
    
  ConstantClass* Create(const TypeClass *Ty, const ValType &V,
                        typename MapTy::iterator I) {
    ConstantClass* Result =
      ConstantCreator<ConstantClass,TypeClass,ValType>::create(Ty, V);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    I = Map.insert(I, std::make_pair(MapKey(Ty, V), Result));

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.insert(std::make_pair(Result, I));

    // If the type of the constant is abstract, make sure that an entry
    // exists for it in the AbstractTypeMap.
    if (Ty->isAbstract()) {
      typename AbstractTypeMapTy::iterator TI = 
                                               AbstractTypeMap.find(Ty);

      if (TI == AbstractTypeMap.end()) {
        // Add ourselves to the ATU list of the type.
        cast<DerivedType>(Ty)->addAbstractTypeUser(this);

        AbstractTypeMap.insert(TI, std::make_pair(Ty, I));
      }
    }
      
    return Result;
  }
public:
    
  /// getOrCreate - Return the specified constant from the map, creating it if
  /// necessary.
  ConstantClass *getOrCreate(const TypeClass *Ty, const ValType &V) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    MapKey Lookup(Ty, V);
    ConstantClass* Result = 0;
    
    typename MapTy::iterator I = Map.find(Lookup);
    // Is it in the map?  
    if (I != Map.end())
      Result = static_cast<ConstantClass *>(I->second);
        
    if (!Result) {
      // If no preexisting value, create one now...
      Result = Create(Ty, V, I);
    }
        
    return Result;
  }

  void remove(ConstantClass *CP) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    typename MapTy::iterator I = FindExistingElement(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->second == CP && "Didn't find correct element?");

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.erase(CP);
      
    // Now that we found the entry, make sure this isn't the entry that
    // the AbstractTypeMap points to.
    const TypeClass *Ty = static_cast<const TypeClass *>(I->first.first);
    if (Ty->isAbstract()) {
      assert(AbstractTypeMap.count(Ty) &&
             "Abstract type not in AbstractTypeMap?");
      typename MapTy::iterator &ATMEntryIt = AbstractTypeMap[Ty];
      if (ATMEntryIt == I) {
        // Yes, we are removing the representative entry for this type.
        // See if there are any other entries of the same type.
        typename MapTy::iterator TmpIt = ATMEntryIt;

        // First check the entry before this one...
        if (TmpIt != Map.begin()) {
          --TmpIt;
          if (TmpIt->first.first != Ty) // Not the same type, move back...
            ++TmpIt;
        }

        // If we didn't find the same type, try to move forward...
        if (TmpIt == ATMEntryIt) {
          ++TmpIt;
          if (TmpIt == Map.end() || TmpIt->first.first != Ty)
            --TmpIt;   // No entry afterwards with the same type
        }

        // If there is another entry in the map of the same abstract type,
        // update the AbstractTypeMap entry now.
        if (TmpIt != ATMEntryIt) {
          ATMEntryIt = TmpIt;
        } else {
          // Otherwise, we are removing the last instance of this type
          // from the table.  Remove from the ATM, and from user list.
          cast<DerivedType>(Ty)->removeAbstractTypeUser(this);
          AbstractTypeMap.erase(Ty);
        }
      }
    }

    Map.erase(I);
  }

    
  /// MoveConstantToNewSlot - If we are about to change C to be the element
  /// specified by I, update our internal data structures to reflect this
  /// fact.
  /// NOTE: This function is not locked. It is the responsibility of the
  /// caller to enforce proper synchronization if using this method.
  void MoveConstantToNewSlot(ConstantClass *C, typename MapTy::iterator I) {
    // First, remove the old location of the specified constant in the map.
    typename MapTy::iterator OldI = FindExistingElement(C);
    assert(OldI != Map.end() && "Constant not found in constant table!");
    assert(OldI->second == C && "Didn't find correct element?");
      
    // If this constant is the representative element for its abstract type,
    // update the AbstractTypeMap so that the representative element is I.
    if (C->getType()->isAbstract()) {
      typename AbstractTypeMapTy::iterator ATI =
          AbstractTypeMap.find(C->getType());
      assert(ATI != AbstractTypeMap.end() &&
             "Abstract type not in AbstractTypeMap?");
      if (ATI->second == OldI)
        ATI->second = I;
    }
      
    // Remove the old entry from the map.
    Map.erase(OldI);
    
    // Update the inverse map so that we know that this constant is now
    // located at descriptor I.
    if (HasLargeKey) {
      assert(I->second == C && "Bad inversemap entry!");
      InverseMap[C] = I;
    }
  }
    
  void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    typename AbstractTypeMapTy::iterator I =
      AbstractTypeMap.find(cast<Type>(OldTy));

    assert(I != AbstractTypeMap.end() &&
           "Abstract type not in AbstractTypeMap?");

    // Convert a constant at a time until the last one is gone.  The last one
    // leaving will remove() itself, causing the AbstractTypeMapEntry to be
    // eliminated eventually.
    do {
      ConvertConstantType<ConstantClass,
                          TypeClass>::convert(
                              static_cast<ConstantClass *>(I->second->second),
                                              cast<TypeClass>(NewTy));

      I = AbstractTypeMap.find(cast<Type>(OldTy));
    } while (I != AbstractTypeMap.end());
  }

  // If the type became concrete without being refined to any other existing
  // type, we just remove ourselves from the ATU list.
  void typeBecameConcrete(const DerivedType *AbsTy) {
    AbsTy->removeAbstractTypeUser(this);
  }

  void dump() const {
    DOUT << "Constant.cpp: ValueMap\n";
  }
};


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
  
  ValueMap<char, Type, ConstantAggregateZero> AggZeroConstants;
  
  typedef ValueMap<std::vector<Constant*>, ArrayType, 
    ConstantArray, true /*largekey*/> ArrayConstantsTy;
  ArrayConstantsTy ArrayConstants;
  
  typedef ValueMap<std::vector<Constant*>, StructType,
                   ConstantStruct, true /*largekey*/> StructConstantsTy;
  StructConstantsTy StructConstants;
  
  typedef ValueMap<std::vector<Constant*>, VectorType,
                   ConstantVector> VectorConstantsTy;
  VectorConstantsTy VectorConstants;
  
  LLVMContext &Context;
  ConstantInt *TheTrueVal;
  ConstantInt *TheFalseVal;
  
  LLVMContextImpl();
  LLVMContextImpl(const LLVMContextImpl&);
  
  friend class ConstantInt;
  friend class ConstantFP;
  friend class ConstantStruct;
  friend class ConstantArray;
  friend class ConstantVector;
public:
  LLVMContextImpl(LLVMContext &C);
  
  MDString *getMDString(const char *StrBegin, unsigned StrLength);
  
  MDNode *getMDNode(Value*const* Vals, unsigned NumVals);
  
  ConstantAggregateZero *getConstantAggregateZero(const Type *Ty);
  
  ConstantInt *getTrue() {
    if (TheTrueVal)
      return TheTrueVal;
    else
      return (TheTrueVal = ConstantInt::get(IntegerType::get(1), 1));
  }
  
  ConstantInt *getFalse() {
    if (TheFalseVal)
      return TheFalseVal;
    else
      return (TheFalseVal = ConstantInt::get(IntegerType::get(1), 0));
  }
  
  void erase(MDString *M);
  void erase(MDNode *M);
  void erase(ConstantAggregateZero *Z);
  void erase(ConstantVector *V);
};

}

#endif
