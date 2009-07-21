//===--------------- LLVMContextImpl.cpp - Implementation ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/MDNode.h"
using namespace llvm;

static char getValType(ConstantAggregateZero *CPZ) { return 0; }

static std::vector<Constant*> getValType(ConstantArray *CA) {
  std::vector<Constant*> Elements;
  Elements.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CA->getOperand(i)));
  return Elements;
}

namespace llvm {
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
    Constant *New = NewTy->getContext().getConstantArray(NewTy, C);
    assert(New != OldC && "Didn't replace constant??");
    OldC->uncheckedReplaceAllUsesWith(New);
    OldC->destroyConstant();    // This constant is now dead, destroy it.
  }
};
}
  
template<class ValType, class TypeClass, class ConstantClass,
         bool HasLargeKey  /*true for arrays and structs*/ >
class VISIBILITY_HIDDEN ValueMap : public AbstractTypeUser {
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

LLVMContextImpl::LLVMContextImpl(LLVMContext &C) :
    Context(C), TheTrueVal(0), TheFalseVal(0) {
  AggZeroConstants = new ValueMap<char, Type, ConstantAggregateZero>();
  ArrayConstants = new ArrayConstantsTy();
}

LLVMContextImpl::~LLVMContextImpl() {
  delete AggZeroConstants;
  delete ArrayConstants;
}

// Get a ConstantInt from an APInt. Note that the value stored in the DenseMap 
// as the key, is a DenseMapAPIntKeyInfo::KeyTy which has provided the
// operator== and operator!= to ensure that the DenseMap doesn't attempt to
// compare APInt's of different widths, which would violate an APInt class
// invariant which generates an assertion.
ConstantInt *LLVMContextImpl::getConstantInt(const APInt& V) {
  // Get the corresponding integer type for the bit width of the value.
  const IntegerType *ITy = Context.getIntegerType(V.getBitWidth());
  // get an existing value or the insertion position
  DenseMapAPIntKeyInfo::KeyTy Key(V, ITy);
  
  ConstantsLock.reader_acquire();
  ConstantInt *&Slot = IntConstants[Key]; 
  ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    ConstantInt *&NewSlot = IntConstants[Key]; 
    if (!Slot) {
      NewSlot = new ConstantInt(ITy, V);
    }
    
    return NewSlot;
  } else {
    return Slot;
  }
}

ConstantFP *LLVMContextImpl::getConstantFP(const APFloat &V) {
  DenseMapAPFloatKeyInfo::KeyTy Key(V);
  
  ConstantsLock.reader_acquire();
  ConstantFP *&Slot = FPConstants[Key];
  ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    ConstantFP *&NewSlot = FPConstants[Key];
    if (!NewSlot) {
      const Type *Ty;
      if (&V.getSemantics() == &APFloat::IEEEsingle)
        Ty = Type::FloatTy;
      else if (&V.getSemantics() == &APFloat::IEEEdouble)
        Ty = Type::DoubleTy;
      else if (&V.getSemantics() == &APFloat::x87DoubleExtended)
        Ty = Type::X86_FP80Ty;
      else if (&V.getSemantics() == &APFloat::IEEEquad)
        Ty = Type::FP128Ty;
      else {
        assert(&V.getSemantics() == &APFloat::PPCDoubleDouble && 
               "Unknown FP format");
        Ty = Type::PPC_FP128Ty;
      }
      NewSlot = new ConstantFP(Ty, V);
    }
    
    return NewSlot;
  }
  
  return Slot;
}

MDString *LLVMContextImpl::getMDString(const char *StrBegin,
                                       const char *StrEnd) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  StringMapEntry<MDString *> &Entry = MDStringCache.GetOrCreateValue(
                                        StrBegin, StrEnd);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Entry.getKeyData(),
                           Entry.getKeyData() + Entry.getKeyLength());

  return S;
}

MDNode *LLVMContextImpl::getMDNode(Value*const* Vals, unsigned NumVals) {
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  ConstantsLock.reader_acquire();
  void *InsertPoint;
  MDNode *N = MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  ConstantsLock.reader_release();
  
  if (!N) {
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    N = MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    if (!N) {
      // InsertPoint will have been set by the FindNodeOrInsertPos call.
      N = new(0) MDNode(Vals, NumVals);
      MDNodeSet.InsertNode(N, InsertPoint);
    }
  }

  return N;
}

ConstantAggregateZero*
LLVMContextImpl::getConstantAggregateZero(const Type *Ty) {
  assert((isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) &&
         "Cannot create an aggregate zero of non-aggregate type!");

  // Implicitly locked.
  return AggZeroConstants->getOrCreate(Ty, 0);
}

Constant *LLVMContextImpl::getConstantArray(const ArrayType *Ty,
                             const std::vector<Constant*> &V) {
  // If this is an all-zero array, return a ConstantAggregateZero object
  if (!V.empty()) {
    Constant *C = V[0];
    if (!C->isNullValue()) {
      // Implicitly locked.
      return ArrayConstants->getOrCreate(Ty, V);
    }
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        // Implicitly locked.
        return ArrayConstants->getOrCreate(Ty, V);
      }
  }
  
  return Context.getConstantAggregateZero(Ty);
}

// *** erase methods ***

void LLVMContextImpl::erase(MDString *M) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  MDStringCache.erase(MDStringCache.find(M->StrBegin, M->StrEnd));
}

void LLVMContextImpl::erase(MDNode *M) {
  sys::SmartScopedWriter<true> Writer(ConstantsLock);
  MDNodeSet.RemoveNode(M);
}

void LLVMContextImpl::erase(ConstantAggregateZero *Z) {
  AggZeroConstants->remove(Z);
}

void LLVMContextImpl::erase(ConstantArray *C) {
  ArrayConstants->remove(C);
}

// *** RAUW helpers ***
Constant *LLVMContextImpl::replaceUsesOfWithOnConstant(ConstantArray *CA,
                                               Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  std::pair<ArrayConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = CA->getType();
  Lookup.second = CA;

  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(CA->getNumOperands());  // Build replacement array.

  // Fill values with the modified operands of the constant array.  Also, 
  // compute whether this turns into an all-zeros array.
  bool isAllZeros = false;
  unsigned NumUpdated = 0;
  if (!ToC->isNullValue()) {
    for (Use *O = CA->OperandList, *E = CA->OperandList + CA->getNumOperands();
         O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
    }
  } else {
    isAllZeros = true;
    for (Use *O = CA->OperandList, *E = CA->OperandList + CA->getNumOperands();
         O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = Context.getConstantAggregateZero(CA->getType());
  } else {
    // Check to see if we have this array type already.
    sys::SmartScopedWriter<true> Writer(ConstantsLock);
    bool Exists;
    ArrayConstantsTy::MapTy::iterator I =
      ArrayConstants->InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant array, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      ArrayConstants->MoveConstantToNewSlot(CA, I);
      
      // Update to the new value.  Optimize for the case when we have a single
      // operand that we're changing, but handle bulk updates efficiently.
      if (NumUpdated == 1) {
        unsigned OperandToUpdate = U - CA->OperandList;
        assert(CA->getOperand(OperandToUpdate) == From &&
               "ReplaceAllUsesWith broken!");
        CA->setOperand(OperandToUpdate, ToC);
      } else {
        for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
          if (CA->getOperand(i) == From)
            CA->setOperand(i, ToC);
      }
      return 0;
    }
  }
  
  return Replacement;
}

