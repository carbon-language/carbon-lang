//===-- llvm/Type.h - Classes for handling data types -----------*- C++ -*-===//
//
// This file contains the declaration of the Type class.  For more "Type" type
// stuff, look in DerivedTypes.h.
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
#include "Support/GraphTraits.h"
#include "Support/iterator"

class DerivedType;
class FunctionType;
class ArrayType;
class PointerType;
class StructType;
class OpaqueType;

struct Type : public Value {
  ///===-------------------------------------------------------------------===//
  /// Definitions of all of the base types for the Type system.  Based on this
  /// value, you can cast to a "DerivedType" subclass (see DerivedTypes.h)
  /// Note: If you add an element to this, you need to add an element to the 
  /// Type::getPrimitiveType function, or else things will break!
  ///
  enum PrimitiveID {
    VoidTyID = 0  , BoolTyID,           //  0, 1: Basics...
    UByteTyID     , SByteTyID,          //  2, 3: 8 bit types...
    UShortTyID    , ShortTyID,          //  4, 5: 16 bit types...
    UIntTyID      , IntTyID,            //  6, 7: 32 bit types...
    ULongTyID     , LongTyID,           //  8, 9: 64 bit types...

    FloatTyID     , DoubleTyID,         // 10,11: Floating point types...

    TypeTyID,                           // 12   : Type definitions
    LabelTyID     ,                     // 13   : Labels... 

    // Derived types... see DerivedTypes.h file...
    // Make sure FirstDerivedTyID stays up to date!!!
    FunctionTyID  , StructTyID,         // Functions... Structs...
    ArrayTyID     , PointerTyID,        // Array... pointer...
    OpaqueTyID,                         // Opaque type instances...
    //PackedTyID  ,                     // SIMD 'packed' format... TODO
    //...

    NumPrimitiveIDs,                    // Must remain as last defined ID
    FirstDerivedTyID = FunctionTyID,
  };

private:
  PrimitiveID ID;        // The current base type of this type...
  unsigned    UID;       // The unique ID number for this class
  bool        Abstract;  // True if type contains an OpaqueType

  const Type *getForwardedTypeInternal() const;
protected:
  /// ctor is protected, so only subclasses can create Type objects...
  Type(const std::string &Name, PrimitiveID id);
  virtual ~Type() {}

  /// setName - Associate the name with this type in the symbol table, but don't
  /// set the local name to be equal specified name.
  ///
  virtual void setName(const std::string &Name, SymbolTable *ST = 0);

  /// Types can become nonabstract later, if they are refined.
  ///
  inline void setAbstract(bool Val) { Abstract = Val; }

  /// isTypeAbstract - This method is used to calculate the Abstract bit.
  ///
  bool isTypeAbstract();

  /// ForwardType - This field is used to implement the union find scheme for
  /// abstract types.  When types are refined to other types, this field is set
  /// to the more refined type.  Only abstract types can be forwarded.
  mutable const Type *ForwardType;

public:
  virtual void print(std::ostream &O) const;

  //===--------------------------------------------------------------------===//
  // Property accessors for dealing with types... Some of these virtual methods
  // are defined in private classes defined in Type.cpp for primitive types.
  //

  /// getPrimitiveID - Return the base type of the type.  This will return one
  /// of the PrimitiveID enum elements defined above.
  ///
  inline PrimitiveID getPrimitiveID() const { return ID; }

  /// getUniqueID - Returns the UID of the type.  This can be thought of as a
  /// small integer version of the pointer to the type class.  Two types that
  /// are structurally different have different UIDs.  This can be used for
  /// indexing types into an array.
  ///
  inline unsigned getUniqueID() const { return UID; }

  /// getDescription - Return the string representation of the type...
  const std::string &getDescription() const;

  /// isSigned - Return whether an integral numeric type is signed.  This is
  /// true for SByteTy, ShortTy, IntTy, LongTy.  Note that this is not true for
  /// Float and Double.
  //
  virtual bool isSigned() const { return 0; }
  
  /// isUnsigned - Return whether a numeric type is unsigned.  This is not quite
  /// the complement of isSigned... nonnumeric types return false as they do
  /// with isSigned.  This returns true for UByteTy, UShortTy, UIntTy, and
  /// ULongTy
  /// 
  virtual bool isUnsigned() const { return 0; }

  /// isInteger - Equilivent to isSigned() || isUnsigned(), but with only a
  /// single virtual function invocation.
  ///
  virtual bool isInteger() const { return 0; }

  /// isIntegral - Returns true if this is an integral type, which is either
  /// BoolTy or one of the Integer types.
  ///
  bool isIntegral() const { return isInteger() || this == BoolTy; }

  /// isFloatingPoint - Return true if this is one of the two floating point
  /// types
  bool isFloatingPoint() const { return ID == FloatTyID || ID == DoubleTyID; }

  /// isAbstract - True if the type is either an Opaque type, or is a derived
  /// type that includes an opaque type somewhere in it.  
  ///
  inline bool isAbstract() const { return Abstract; }

  /// isLosslesslyConvertibleTo - Return true if this type can be converted to
  /// 'Ty' without any reinterpretation of bits.  For example, uint to int.
  ///
  bool isLosslesslyConvertibleTo(const Type *Ty) const;


  /// Here are some useful little methods to query what type derived types are
  /// Note that all other types can just compare to see if this == Type::xxxTy;
  ///
  inline bool isPrimitiveType() const { return ID < FirstDerivedTyID;  }
  inline bool isDerivedType()   const { return ID >= FirstDerivedTyID; }

  /// isFirstClassType - Return true if the value is holdable in a register.
  inline bool isFirstClassType() const {
    return isPrimitiveType() || ID == PointerTyID;
  }

  /// isSized - Return true if it makes sense to take the size of this type.  To
  /// get the actual size for a particular target, it is reasonable to use the
  /// TargetData subsystem to do this.
  ///
  bool isSized() const {
    return ID != VoidTyID && ID != TypeTyID &&
           ID != FunctionTyID && ID != LabelTyID && ID != OpaqueTyID;
  }

  /// getPrimitiveSize - Return the basic size of this type if it is a primative
  /// type.  These are fixed by LLVM and are not target dependent.  This will
  /// return zero if the type does not have a size or is not a primitive type.
  ///
  unsigned getPrimitiveSize() const;

  /// getForwaredType - Return the type that this type has been resolved to if
  /// it has been resolved to anything.  This is used to implement the
  /// union-find algorithm for type resolution.
  const Type *getForwardedType() const {
    if (!ForwardType) return 0;
    return getForwardedTypeInternal();
  }

  //===--------------------------------------------------------------------===//
  // Type Iteration support
  //
  class TypeIterator;
  typedef TypeIterator subtype_iterator;
  inline subtype_iterator subtype_begin() const;   // DEFINED BELOW
  inline subtype_iterator subtype_end() const;     // DEFINED BELOW

  /// getContainedType - This method is used to implement the type iterator
  /// (defined a the end of the file).  For derived types, this returns the
  /// types 'contained' in the derived type.
  ///
  virtual const Type *getContainedType(unsigned i) const {
    assert(0 && "No contained types!");
  }

  /// getNumContainedTypes - Return the number of types in the derived type
  virtual unsigned getNumContainedTypes() const { return 0; }

  //===--------------------------------------------------------------------===//
  // Static members exported by the Type class itself.  Useful for getting
  // instances of Type.
  //

  /// getPrimitiveType/getUniqueIDType - Return a type based on an identifier.
  static const Type *getPrimitiveType(PrimitiveID IDNumber);
  static const Type *getUniqueIDType(unsigned UID);

  //===--------------------------------------------------------------------===//
  // These are the builtin types that are always available...
  //
  static Type *VoidTy , *BoolTy;
  static Type *SByteTy, *UByteTy,
              *ShortTy, *UShortTy,
              *IntTy  , *UIntTy, 
              *LongTy , *ULongTy;
  static Type *FloatTy, *DoubleTy;

  static Type *TypeTy , *LabelTy;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Type *T) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::TypeVal;
  }

#include "llvm/Type.def"

private:
  class TypeIterator : public bidirectional_iterator<const Type, ptrdiff_t> {
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


// Provide specializations of GraphTraits to be able to treat a type as a 
// graph of sub types...

template <> struct GraphTraits<Type*> {
  typedef Type NodeType;
  typedef Type::subtype_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(Type *T) { return T; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->subtype_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->subtype_end();
  }
};

template <> struct GraphTraits<const Type*> {
  typedef const Type NodeType;
  typedef Type::subtype_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(const Type *T) { return T; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->subtype_begin(); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->subtype_end();
  }
};

template <> inline bool isa_impl<PointerType, Type>(const Type &Ty) { 
  return Ty.getPrimitiveID() == Type::PointerTyID;
}

#endif
