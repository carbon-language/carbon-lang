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
// Opaque types are simple derived types with no state.  There may be many
// different Opaque type objects floating around, but two are only considered
// identical if they are pointer equals of each other.  This allows us to have 
// two opaque types that end up resolving to different concrete types later.
//
// Opaque types are also kinda wierd and scary and different because they have
// to keep a list of uses of the type.  When, through linking, parsing, or
// bytecode reading, they become resolved, they need to find and update all
// users of the unknown type, causing them to reference a new, more concrete
// type.  Opaque types are deleted when their use list dwindles to zero users.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPE_H
#define LLVM_TYPE_H

#include "llvm/Value.h"

class DerivedType;
class MethodType;
class ArrayType;
class PointerType;
class StructType;
class OpaqueType;

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
    LabelTyID     ,                     // 13   : Labels... 
    /*LockTyID , */                     // 14   : mutex - TODO

    // Derived types... see DerivedTypes.h file...
    // Make sure FirstDerivedTyID stays up to date!!!
    MethodTyID    , ModuleTyID,         // Methods... Modules...
    ArrayTyID     , PointerTyID,        // Array... pointer...
    StructTyID    , OpaqueTyID,         // Structure... Opaque type instances...
    //PackedTyID  ,                     // SIMD 'packed' format... TODO
    //...

    NumPrimitiveIDs,                    // Must remain as last defined ID
    FirstDerivedTyID = MethodTyID,
  };

private:
  PrimitiveID ID;        // The current base type of this type...
  unsigned    UID;       // The unique ID number for this class
  string      Desc;      // The printed name of the string...
  bool        Abstract;  // True if type contains an OpaqueType
  bool        Recursive; // True if the type is recursive

protected:
  // ctor is protected, so only subclasses can create Type objects...
  Type(const string &Name, PrimitiveID id);
  virtual ~Type() {}

  // When types are refined, they update their description to be more concrete.
  //
  inline void setDescription(const string &D) { Desc = D; }
  
  // setName - Associate the name with this type in the symbol table, but don't
  // set the local name to be equal specified name.
  //
  virtual void setName(const string &Name, SymbolTable *ST = 0);

  // Types can become nonabstract later, if they are refined.
  //
  inline void setAbstract(bool Val) { Abstract = Val; }

  // Types can become recursive later, if they are refined.
  //
  inline void setRecursive(bool Val) { Recursive = Val; }

public:

  //===--------------------------------------------------------------------===//
  // Property accessors for dealing with types...
  //

  // getPrimitiveID - Return the base type of the type.  This will return one
  // of the PrimitiveID enum elements defined above.
  //
  inline PrimitiveID getPrimitiveID() const { return ID; }

  // getUniqueID - Returns the UID of the type.  This can be thought of as a 
  // small integer version of the pointer to the type class.  Two types that are
  // structurally different have different UIDs.  This can be used for indexing
  // types into an array.
  //
  inline unsigned getUniqueID() const { return UID; }

  // getDescription - Return the string representation of the type...
  inline const string &getDescription() const { return Desc; }

  // isSigned - Return whether a numeric type is signed.
  virtual bool isSigned() const { return 0; }
  
  // isUnsigned - Return whether a numeric type is unsigned.  This is not 
  // quite the complement of isSigned... nonnumeric types return false as they
  // do with isSigned.
  // 
  virtual bool isUnsigned() const { return 0; }

  // isIntegral - Equilivent to isSigned() || isUnsigned, but with only a single
  // virtual function invocation.
  //
  virtual bool isIntegral() const { return 0; }

  // isAbstract - True if the type is either an Opaque type, or is a derived
  // type that includes an opaque type somewhere in it.  
  //
  inline bool isAbstract() const { return Abstract; }

  // isRecursive - True if the type graph contains a cycle.
  //
  inline bool isRecursive() const { return Recursive; }

  //===--------------------------------------------------------------------===//
  // Type Iteration support
  //
  class TypeIterator;
  typedef TypeIterator subtype_iterator;
  inline subtype_iterator subtype_begin() const;   // DEFINED BELOW
  inline subtype_iterator subtype_end() const;     // DEFINED BELOW

  // getContainedType - This method is used to implement the type iterator
  // (defined a the end of the file).  For derived types, this returns the types
  // 'contained' in the derived type, returning 0 when 'i' becomes invalid. This
  // allows the user to iterate over the types in a struct, for example, really
  // easily.
  //
  virtual const Type *getContainedType(unsigned i) const { return 0; }

  // getNumContainedTypes - Return the number of types in the derived type
  virtual unsigned getNumContainedTypes() const { return 0; }

  //===--------------------------------------------------------------------===//
  // Static members exported by the Type class itself.  Useful for getting
  // instances of Type.
  //

  // getPrimitiveType/getUniqueIDType - Return a type based on an identifier.
  static const Type *getPrimitiveType(PrimitiveID IDNumber);
  static const Type *getUniqueIDType(unsigned UID);

  //===--------------------------------------------------------------------===//
  // These are the builtin types that are always available...
  //
  static const Type *VoidTy , *BoolTy;
  static const Type *SByteTy, *UByteTy,
                    *ShortTy, *UShortTy,
                    *IntTy  , *UIntTy, 
                    *LongTy , *ULongTy;
  static const Type *FloatTy, *DoubleTy;

  static const Type *TypeTy , *LabelTy; //, *LockTy;

  // Here are some useful little methods to query what type derived types are
  // Note that all other types can just compare to see if this == Type::xxxTy;
  //
  inline bool isDerivedType()   const { return ID >= FirstDerivedTyID; }
  inline bool isPrimitiveType() const { return ID < FirstDerivedTyID;  }

  inline bool isLabelType()     const { return this == LabelTy; }

  inline const DerivedType *castDerivedType() const {
    return isDerivedType() ? (const DerivedType*)this : 0;
  }
  inline const DerivedType *castDerivedTypeAsserting() const {
    assert(isDerivedType());
    return (const DerivedType*)this;
  }

  inline const MethodType *isMethodType() const {
    return ID == MethodTyID ? (const MethodType*)this : 0;
  }
  inline bool isModuleType()    const { return ID == ModuleTyID;     }
  inline const ArrayType *isArrayType() const { 
    return ID == ArrayTyID ? (const ArrayType*)this : 0;
  }
  inline const PointerType *isPointerType() const { 
    return ID == PointerTyID ? (const PointerType*)this : 0;
  }
  inline const StructType *isStructType() const {
    return ID == StructTyID ? (const StructType*)this : 0;
  }
  inline const OpaqueType *isOpaqueType() const {
    return ID == OpaqueTyID ? (const OpaqueType*)this : 0;
  }

private:
  class TypeIterator : public std::bidirectional_iterator<const Type,
		                                          ptrdiff_t> {
    const Type * const Ty;
    unsigned Idx;

    typedef TypeIterator _Self;
  public:
    inline TypeIterator(const Type *ty, unsigned idx) : Ty(ty), Idx(idx) {}
    inline ~TypeIterator() {}
    
    inline bool operator==(const _Self& x) const { return Idx == x.Idx; }
    inline bool operator!=(const _Self& x) const { return !operator==(x); }
    
    inline pointer operator*() const { return Ty->getContainedType(Idx); }
    inline pointer operator->() const { return operator*(); }
    
    inline _Self& operator++() { ++Idx; return *this; } // Preincrement
    inline _Self operator++(int) { // Postincrement
      _Self tmp = *this; ++*this; return tmp; 
    }
    
    inline _Self& operator--() { --Idx; return *this; }  // Predecrement
    inline _Self operator--(int) { // Postdecrement
      _Self tmp = *this; --*this; return tmp;
    }
  };
};

inline Type::TypeIterator Type::subtype_begin() const {
  return TypeIterator(this, 0);
}

inline Type::TypeIterator Type::subtype_end() const {
  return TypeIterator(this, getNumContainedTypes());
}

#endif
