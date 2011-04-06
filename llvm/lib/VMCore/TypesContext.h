//===-- TypesContext.h - Types-related Context Internals ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines various helper methods and classes used by
// LLVMContextImpl for creating and managing types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPESCONTEXT_H
#define LLVM_TYPESCONTEXT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <map>


//===----------------------------------------------------------------------===//
//                       Derived Type Factory Functions
//===----------------------------------------------------------------------===//
namespace llvm {

/// getSubElementHash - Generate a hash value for all of the SubType's of this
/// type.  The hash value is guaranteed to be zero if any of the subtypes are 
/// an opaque type.  Otherwise we try to mix them in as well as possible, but do
/// not look at the subtype's subtype's.
static unsigned getSubElementHash(const Type *Ty) {
  unsigned HashVal = 0;
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I) {
    HashVal *= 32;
    const Type *SubTy = I->get();
    HashVal += SubTy->getTypeID();
    switch (SubTy->getTypeID()) {
    default: break;
    case Type::OpaqueTyID: return 0;    // Opaque -> hash = 0 no matter what.
    case Type::IntegerTyID:
      HashVal ^= (cast<IntegerType>(SubTy)->getBitWidth() << 3);
      break;
    case Type::FunctionTyID:
      HashVal ^= cast<FunctionType>(SubTy)->getNumParams()*2 + 
                 cast<FunctionType>(SubTy)->isVarArg();
      break;
    case Type::ArrayTyID:
      HashVal ^= cast<ArrayType>(SubTy)->getNumElements();
      break;
    case Type::VectorTyID:
      HashVal ^= cast<VectorType>(SubTy)->getNumElements();
      break;
    case Type::StructTyID:
      HashVal ^= cast<StructType>(SubTy)->getNumElements();
      break;
    case Type::PointerTyID:
      HashVal ^= cast<PointerType>(SubTy)->getAddressSpace();
      break;
    }
  }
  return HashVal ? HashVal : 1;  // Do not return zero unless opaque subty.
}

//===----------------------------------------------------------------------===//
// Integer Type Factory...
//
class IntegerValType {
  uint32_t bits;
public:
  IntegerValType(uint32_t numbits) : bits(numbits) {}

  static IntegerValType get(const IntegerType *Ty) {
    return IntegerValType(Ty->getBitWidth());
  }

  static unsigned hashTypeStructure(const IntegerType *Ty) {
    return (unsigned)Ty->getBitWidth();
  }

  inline bool operator<(const IntegerValType &IVT) const {
    return bits < IVT.bits;
  }
};

// PointerValType - Define a class to hold the key that goes into the TypeMap
//
class PointerValType {
  const Type *ValTy;
  unsigned AddressSpace;
public:
  PointerValType(const Type *val, unsigned as) : ValTy(val), AddressSpace(as) {}

  static PointerValType get(const PointerType *PT) {
    return PointerValType(PT->getElementType(), PT->getAddressSpace());
  }

  static unsigned hashTypeStructure(const PointerType *PT) {
    return getSubElementHash(PT);
  }

  bool operator<(const PointerValType &MTV) const {
    if (AddressSpace < MTV.AddressSpace) return true;
    return AddressSpace == MTV.AddressSpace && ValTy < MTV.ValTy;
  }
};

//===----------------------------------------------------------------------===//
// Array Type Factory...
//
class ArrayValType {
  const Type *ValTy;
  uint64_t Size;
public:
  ArrayValType(const Type *val, uint64_t sz) : ValTy(val), Size(sz) {}

  static ArrayValType get(const ArrayType *AT) {
    return ArrayValType(AT->getElementType(), AT->getNumElements());
  }

  static unsigned hashTypeStructure(const ArrayType *AT) {
    return (unsigned)AT->getNumElements();
  }

  inline bool operator<(const ArrayValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy < MTV.ValTy;
  }
};

//===----------------------------------------------------------------------===//
// Vector Type Factory...
//
class VectorValType {
  const Type *ValTy;
  unsigned Size;
public:
  VectorValType(const Type *val, int sz) : ValTy(val), Size(sz) {}

  static VectorValType get(const VectorType *PT) {
    return VectorValType(PT->getElementType(), PT->getNumElements());
  }

  static unsigned hashTypeStructure(const VectorType *PT) {
    return PT->getNumElements();
  }

  inline bool operator<(const VectorValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy < MTV.ValTy;
  }
};

// StructValType - Define a class to hold the key that goes into the TypeMap
//
class StructValType {
  std::vector<const Type*> ElTypes;
  bool packed;
public:
  StructValType(ArrayRef<const Type*> args, bool isPacked)
    : ElTypes(args.vec()), packed(isPacked) {}

  static StructValType get(const StructType *ST) {
    std::vector<const Type *> ElTypes;
    ElTypes.reserve(ST->getNumElements());
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i)
      ElTypes.push_back(ST->getElementType(i));

    return StructValType(ElTypes, ST->isPacked());
  }

  static unsigned hashTypeStructure(const StructType *ST) {
    return ST->getNumElements();
  }

  inline bool operator<(const StructValType &STV) const {
    if (ElTypes < STV.ElTypes) return true;
    else if (ElTypes > STV.ElTypes) return false;
    else return (int)packed < (int)STV.packed;
  }
};

// FunctionValType - Define a class to hold the key that goes into the TypeMap
//
class FunctionValType {
  const Type *RetTy;
  std::vector<const Type*> ArgTypes;
  bool isVarArg;
public:
  FunctionValType(const Type *ret, ArrayRef<const Type*> args, bool isVA)
    : RetTy(ret), ArgTypes(args.vec()), isVarArg(isVA) {}

  static FunctionValType get(const FunctionType *FT);

  static unsigned hashTypeStructure(const FunctionType *FT) {
    unsigned Result = FT->getNumParams()*2 + FT->isVarArg();
    return Result;
  }

  inline bool operator<(const FunctionValType &MTV) const {
    if (RetTy < MTV.RetTy) return true;
    if (RetTy > MTV.RetTy) return false;
    if (isVarArg < MTV.isVarArg) return true;
    if (isVarArg > MTV.isVarArg) return false;
    if (ArgTypes < MTV.ArgTypes) return true;
    if (ArgTypes > MTV.ArgTypes) return false;
    return false;
  }
};

class TypeMapBase {
protected:
  /// TypesByHash - Keep track of types by their structure hash value.  Note
  /// that we only keep track of types that have cycles through themselves in
  /// this map.
  ///
  std::multimap<unsigned, PATypeHolder> TypesByHash;

  ~TypeMapBase() {
    // PATypeHolder won't destroy non-abstract types.
    // We can't destroy them by simply iterating, because
    // they may contain references to each-other.
    for (std::multimap<unsigned, PATypeHolder>::iterator I
         = TypesByHash.begin(), E = TypesByHash.end(); I != E; ++I) {
      Type *Ty = const_cast<Type*>(I->second.Ty);
      I->second.destroy();
      // We can't invoke destroy or delete, because the type may
      // contain references to already freed types.
      // So we have to destruct the object the ugly way.
      if (Ty) {
        Ty->AbstractTypeUsers.clear();
        static_cast<const Type*>(Ty)->Type::~Type();
        operator delete(Ty);
      }
    }
  }

public:
  void RemoveFromTypesByHash(unsigned Hash, const Type *Ty) {
    std::multimap<unsigned, PATypeHolder>::iterator I =
      TypesByHash.lower_bound(Hash);
    for (; I != TypesByHash.end() && I->first == Hash; ++I) {
      if (I->second == Ty) {
        TypesByHash.erase(I);
        return;
      }
    }

    // This must be do to an opaque type that was resolved.  Switch down to hash
    // code of zero.
    assert(Hash && "Didn't find type entry!");
    RemoveFromTypesByHash(0, Ty);
  }

  /// TypeBecameConcrete - When Ty gets a notification that TheType just became
  /// concrete, drop uses and make Ty non-abstract if we should.
  void TypeBecameConcrete(DerivedType *Ty, const DerivedType *TheType) {
    // If the element just became concrete, remove 'ty' from the abstract
    // type user list for the type.  Do this for as many times as Ty uses
    // OldType.
    for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
         I != E; ++I)
      if (I->get() == TheType)
        TheType->removeAbstractTypeUser(Ty);

    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!  Note that this
    // may send out notifications to AbstractTypeUsers that types become
    // concrete.
    if (Ty->isAbstract())
      Ty->PromoteAbstractToConcrete();
  }
};

// TypeMap - Make sure that only one instance of a particular type may be
// created on any given run of the compiler... note that this involves updating
// our map if an abstract type gets refined somehow.
//
template<class ValType, class TypeClass>
class TypeMap : public TypeMapBase {
  std::map<ValType, PATypeHolder> Map;
public:
  typedef typename std::map<ValType, PATypeHolder>::iterator iterator;

  inline TypeClass *get(const ValType &V) {
    iterator I = Map.find(V);
    return I != Map.end() ? cast<TypeClass>((Type*)I->second.get()) : 0;
  }

  inline void add(const ValType &V, TypeClass *Ty) {
    Map.insert(std::make_pair(V, Ty));

    // If this type has a cycle, remember it.
    TypesByHash.insert(std::make_pair(ValType::hashTypeStructure(Ty), Ty));
    print("add");
  }
  
  /// RefineAbstractType - This method is called after we have merged a type
  /// with another one.  We must now either merge the type away with
  /// some other type or reinstall it in the map with it's new configuration.
  void RefineAbstractType(TypeClass *Ty, const DerivedType *OldType,
                        const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
    DEBUG(dbgs() << "RefineAbstractType(" << (void*)OldType << "[" << *OldType
                 << "], " << (void*)NewType << " [" << *NewType << "])\n");
#endif
    
    // Otherwise, we are changing one subelement type into another.  Clearly the
    // OldType must have been abstract, making us abstract.
    assert(Ty->isAbstract() && "Refining a non-abstract type!");
    assert(OldType != NewType);

    // Make a temporary type holder for the type so that it doesn't disappear on
    // us when we erase the entry from the map.
    PATypeHolder TyHolder = Ty;

    // The old record is now out-of-date, because one of the children has been
    // updated.  Remove the obsolete entry from the map.
    unsigned NumErased = Map.erase(ValType::get(Ty));
    assert(NumErased && "Element not found!"); (void)NumErased;

    // Remember the structural hash for the type before we start hacking on it,
    // in case we need it later.
    unsigned OldTypeHash = ValType::hashTypeStructure(Ty);

    // Find the type element we are refining... and change it now!
    for (unsigned i = 0, e = Ty->getNumContainedTypes(); i != e; ++i)
      if (Ty->ContainedTys[i] == OldType)
        Ty->ContainedTys[i] = NewType;
    unsigned NewTypeHash = ValType::hashTypeStructure(Ty);
    
    // If there are no cycles going through this node, we can do a simple,
    // efficient lookup in the map, instead of an inefficient nasty linear
    // lookup.
    if (!TypeHasCycleThroughItself(Ty)) {
      typename std::map<ValType, PATypeHolder>::iterator I;
      bool Inserted;

      tie(I, Inserted) = Map.insert(std::make_pair(ValType::get(Ty), Ty));
      if (!Inserted) {
        // Refined to a different type altogether?
        RemoveFromTypesByHash(OldTypeHash, Ty);

        // We already have this type in the table.  Get rid of the newly refined
        // type.
        TypeClass *NewTy = cast<TypeClass>((Type*)I->second.get());
        Ty->refineAbstractTypeTo(NewTy);
        return;
      }
    } else {
      // Now we check to see if there is an existing entry in the table which is
      // structurally identical to the newly refined type.  If so, this type
      // gets refined to the pre-existing type.
      //
      std::multimap<unsigned, PATypeHolder>::iterator I, E, Entry;
      tie(I, E) = TypesByHash.equal_range(NewTypeHash);
      Entry = E;
      for (; I != E; ++I) {
        if (I->second == Ty) {
          // Remember the position of the old type if we see it in our scan.
          Entry = I;
          continue;
        }
        
        if (!TypesEqual(Ty, I->second))
          continue;
        
        TypeClass *NewTy = cast<TypeClass>((Type*)I->second.get());

        // Remove the old entry form TypesByHash.  If the hash values differ
        // now, remove it from the old place.  Otherwise, continue scanning
        // withing this hashcode to reduce work.
        if (NewTypeHash != OldTypeHash) {
          RemoveFromTypesByHash(OldTypeHash, Ty);
        } else {
          if (Entry == E) {
            // Find the location of Ty in the TypesByHash structure if we
            // haven't seen it already.
            while (I->second != Ty) {
              ++I;
              assert(I != E && "Structure doesn't contain type??");
            }
            Entry = I;
          }
          TypesByHash.erase(Entry);
        }
        Ty->refineAbstractTypeTo(NewTy);
        return;
      }

      // If there is no existing type of the same structure, we reinsert an
      // updated record into the map.
      Map.insert(std::make_pair(ValType::get(Ty), Ty));
    }

    // If the hash codes differ, update TypesByHash
    if (NewTypeHash != OldTypeHash) {
      RemoveFromTypesByHash(OldTypeHash, Ty);
      TypesByHash.insert(std::make_pair(NewTypeHash, Ty));
    }
    
    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!  Note that this
    // may send out notifications to AbstractTypeUsers that types become
    // concrete.
    if (Ty->isAbstract())
      Ty->PromoteAbstractToConcrete();
  }

  void print(const char *Arg) const {
#ifdef DEBUG_MERGE_TYPES
    DEBUG(dbgs() << "TypeMap<>::" << Arg << " table contents:\n");
    unsigned i = 0;
    for (typename std::map<ValType, PATypeHolder>::const_iterator I
           = Map.begin(), E = Map.end(); I != E; ++I)
      DEBUG(dbgs() << " " << (++i) << ". " << (void*)I->second.get() << " "
                   << *I->second.get() << "\n");
#endif
  }

  void dump() const { print("dump output"); }
};
}

#endif
