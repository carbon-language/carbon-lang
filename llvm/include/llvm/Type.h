//===-- llvm/Type.h - Classes for handling data types ------------*- C++ -*--=//
//
// This file contains the declaration of the Type class.  For more "Type" type
// stuff, look in DerivedTypes.h and Opt/ConstantHandling.h
//
// Note that instances of the Type class are immutable: once they are created,
// they are never changed.  Also note that only one instance of a particular 
// type is ever created.  Thus seeing if two types are equal is a matter of 
// doing a trivial pointer comparison.
//
// Types, once allocated, are never free'd.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPE_H
#define LLVM_TYPE_H

#include "llvm/Value.h"

namespace opt {
  class ConstRules;
}
class ConstPoolVal;

class Type : public Value {
public:
  //===--------------------------------------------------------------------===//
  // Definitions of all of the base types for the Type system.  Based on this
  // value, you can cast to a "DerivedType" subclass (see DerivedTypes.h)
  // Note: If you add an element to this, you need to add an element to the 
  // Type::getPrimitiveType function, or else things will break!
  //
  enum PrimitiveID {
    VoidTyID = 0  , BoolTyID,           //  0, 1: Basics...
    UByteTyID     , SByteTyID,          //  2, 3: 8 bit types...
    UShortTyID    , ShortTyID,          //  4, 5: 16 bit types...
    UIntTyID      , IntTyID,            //  6, 7: 32 bit types...
    ULongTyID     , LongTyID,           //  8, 9: 64 bit types...

    FloatTyID     , DoubleTyID,         // 10,11: Floating point types...

    TypeTyID,                           // 12   : Type definitions
    LabelTyID     , LockTyID,           // 13,14: Labels... mutexes...

    // TODO: Kill FillerTyID.  It just makes FirstDerivedTyID = 0x10
    FillerTyID ,                        // 15   : filler

    // Derived types... see DerivedTypes.h file...
    // Make sure FirstDerivedTyID stays up to date!!!
    MethodTyID    , ModuleTyID,         // Methods... Modules...
    ArrayTyID     , PointerTyID,        // Array... pointer...
    StructTyID    , PackedTyID,         // Structure... SIMD 'packed' format...
    //...

    NumPrimitiveIDs,                    // Must remain as last defined ID
    FirstDerivedTyID = MethodTyID,
  };

private:
  PrimitiveID ID;    // The current base type of this type...
  unsigned    UID;   // The unique ID number for this class

  // ConstRulesImpl - See Opt/ConstantHandling.h for more info
  mutable const opt::ConstRules *ConstRulesImpl;

protected:
  // ctor is protected, so only subclasses can create Type objects...
  Type(const string &Name, PrimitiveID id);
public:
  virtual ~Type() {}

  // isSigned - Return whether a numeric type is signed.
  virtual bool isSigned() const { return 0; }
  
  // isUnsigned - Return whether a numeric type is unsigned.  This is not 
  // quite the complement of isSigned... nonnumeric types return false as they
  // do with isSigned.
  // 
  virtual bool isUnsigned() const { return 0; }
  
  inline unsigned getUniqueID() const { return UID; }
  inline PrimitiveID getPrimitiveID() const { return ID; }

  // getPrimitiveType/getUniqueIDType - Return a type based on an identifier.
  static const Type *getPrimitiveType(PrimitiveID IDNumber);
  static const Type *getUniqueIDType(unsigned UID);

  // Methods for dealing with constants uniformly.  See Opt/ConstantHandling.h
  // for more info on this...
  //
  inline const opt::ConstRules *getConstRules() const { return ConstRulesImpl; }
  inline void setConstRules(const opt::ConstRules *R) const { ConstRulesImpl = R; }

public:   // These are the builtin types that are always available...
  static const Type *VoidTy , *BoolTy;
  static const Type *SByteTy, *UByteTy,
                    *ShortTy, *UShortTy,
                    *IntTy  , *UIntTy, 
                    *LongTy , *ULongTy;
  static const Type *FloatTy, *DoubleTy;

  static const Type *TypeTy , *LabelTy, *LockTy;

  // Here are some useful little methods to query what type derived types are
  // Note that all other types can just compare to see if this == Type::xxxTy;
  //
  inline bool isDerivedType()   const { return ID >= FirstDerivedTyID; }
  inline bool isPrimitiveType() const { return ID < FirstDerivedTyID;  }

  inline bool isLabelType()     const { return this == LabelTy; }
  inline bool isMethodType()    const { return ID == MethodTyID;     }
  inline bool isModuleType()    const { return ID == ModuleTyID;     }
  inline bool isArrayType()     const { return ID == ArrayTyID;      }
  inline bool isPointerType()   const { return ID == PointerTyID;    }
  inline bool isStructType()    const { return ID == StructTyID;     }
};

#endif
