//===-- llvm/Type.h - Classes for handling data types -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include <vector>

namespace llvm {

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
  enum TypeID {
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

    NumTypeIDs,                         // Must remain as last defined ID
    FirstDerivedTyID = FunctionTyID,
  };

private:
  TypeID   ID;        // The current base type of this type...
  unsigned UID;       // The unique ID number for this class
  bool     Abstract;  // True if type contains an OpaqueType

  /// RefCount - This counts the number of PATypeHolders that are pointing to
  /// this type.  When this number falls to zero, if the type is abstract and
  /// has no AbstractTypeUsers, the type is deleted.  This is only sensical for
  /// derived types.
  ///
  mutable unsigned RefCount;

  const Type *getForwardedTypeInternal() const;
protected:
  /// ctor is protected, so only subclasses can create Type objects...
  Type(const std::string &Name, TypeID id);
  virtual ~Type() {}


  /// Types can become nonabstract later, if they are refined.
  ///
  inline void setAbstract(bool Val) { Abstract = Val; }

  /// isTypeAbstract - This method is used to calculate the Abstract bit.
  ///
  bool isTypeAbstract();

  unsigned getRefCount() const { return RefCount; }

  /// ForwardType - This field is used to implement the union find scheme for
  /// abstract types.  When types are refined to other types, this field is set
  /// to the more refined type.  Only abstract types can be forwarded.
  mutable const Type *ForwardType;

  /// ContainedTys - The list of types contained by this one.  For example, this
  /// includes the arguments of a function type, the elements of the structure,
  /// the pointee of a pointer, etc.  Note that keeping this vector in the Type
  /// class wastes some space for types that do not contain anything (such as
  /// primitive types).  However, keeping it here allows the subtype_* members
  /// to be implemented MUCH more efficiently, and dynamically very few types do
  /// not contain any elements (most are derived).
  std::vector<PATypeHandle> ContainedTys;

public:
  virtual void print(std::ostream &O) const;

  /// @brief Debugging support: print to stderr
  virtual void dump() const;

  /// setName - Associate the name with this type in the symbol table, but don't
  /// set the local name to be equal specified name.
  ///
  virtual void setName(const std::string &Name, SymbolTable *ST = 0);

  //===--------------------------------------------------------------------===//
  // Property accessors for dealing with types... Some of these virtual methods
  // are defined in private classes defined in Type.cpp for primitive types.
  //

  /// getTypeID - Return the type id for the type.  This will return one
  /// of the TypeID enum elements defined above.
  ///
  inline TypeID getTypeID() const { return ID; }

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
  ///
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
    return (ID != VoidTyID && ID < TypeTyID) || ID == PointerTyID;
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

  /// getUnsignedVersion - If this is an integer type, return the unsigned
  /// variant of this type.  For example int -> uint.
  const Type *getUnsignedVersion() const;

  /// getSignedVersion - If this is an integer type, return the signed variant
  /// of this type.  For example uint -> int.
  const Type *getSignedVersion() const;

  /// getForwaredType - Return the type that this type has been resolved to if
  /// it has been resolved to anything.  This is used to implement the
  /// union-find algorithm for type resolution, and shouldn't be used by general
  /// purpose clients.
  const Type *getForwardedType() const {
    if (!ForwardType) return 0;
    return getForwardedTypeInternal();
  }

  //===--------------------------------------------------------------------===//
  // Type Iteration support
  //
  typedef std::vector<PATypeHandle>::const_iterator subtype_iterator;
  subtype_iterator subtype_begin() const { return ContainedTys.begin(); }
  subtype_iterator subtype_end() const { return ContainedTys.end(); }

  /// getContainedType - This method is used to implement the type iterator
  /// (defined a the end of the file).  For derived types, this returns the
  /// types 'contained' in the derived type.
  ///
  const Type *getContainedType(unsigned i) const {
    assert(i < ContainedTys.size() && "Index out of range!");
    return ContainedTys[i];
  }

  /// getNumContainedTypes - Return the number of types in the derived type.
  ///
  unsigned getNumContainedTypes() const { return ContainedTys.size(); }

  //===--------------------------------------------------------------------===//
  // Static members exported by the Type class itself.  Useful for getting
  // instances of Type.
  //

  /// getPrimitiveType/getUniqueIDType - Return a type based on an identifier.
  static const Type *getPrimitiveType(TypeID IDNumber);
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

  // Virtual methods used by callbacks below.  These should only be implemented
  // in the DerivedType class.
  virtual void addAbstractTypeUser(AbstractTypeUser *U) const {
    abort(); // Only on derived types!
  }
  virtual void removeAbstractTypeUser(AbstractTypeUser *U) const {
    abort(); // Only on derived types!
  }

  void addRef() const {
    assert(isAbstract() && "Cannot add a reference to a non-abstract type!");
    ++RefCount;
  }
  
  void dropRef() const {
    assert(isAbstract() && "Cannot drop a refernce to a non-abstract type!");
    assert(RefCount && "No objects are currently referencing this object!");

    // If this is the last PATypeHolder using this object, and there are no
    // PATypeHandles using it, the type is dead, delete it now.
    if (--RefCount == 0)
      RefCountIsZero();
  }
private:
  virtual void RefCountIsZero() const {
    abort(); // only on derived types!
  }

};

//===----------------------------------------------------------------------===//
// Define some inline methods for the AbstractTypeUser.h:PATypeHandle class.
// These are defined here because they MUST be inlined, yet are dependent on 
// the definition of the Type class.  Of course Type derives from Value, which
// contains an AbstractTypeUser instance, so there is no good way to factor out
// the code.  Hence this bit of uglyness.
//
// In the long term, Type should not derive from Value, allowing
// AbstractTypeUser.h to #include Type.h, allowing us to eliminate this
// nastyness entirely.
//
inline void PATypeHandle::addUser() {
  assert(Ty && "Type Handle has a null type!");
  if (Ty->isAbstract())
    Ty->addAbstractTypeUser(User);
}
inline void PATypeHandle::removeUser() {
  if (Ty->isAbstract())
    Ty->removeAbstractTypeUser(User);
}

inline void PATypeHandle::removeUserFromConcrete() {
  if (!Ty->isAbstract())
    Ty->removeAbstractTypeUser(User);
}

// Define inline methods for PATypeHolder...

inline void PATypeHolder::addRef() {
  if (Ty->isAbstract())
    Ty->addRef();
}

inline void PATypeHolder::dropRef() {
  if (Ty->isAbstract())
    Ty->dropRef();
}

/// get - This implements the forwarding part of the union-find algorithm for
/// abstract types.  Before every access to the Type*, we check to see if the
/// type we are pointing to is forwarding to a new type.  If so, we drop our
/// reference to the type.
///
inline const Type* PATypeHolder::get() const {
  const Type *NewTy = Ty->getForwardedType();
  if (!NewTy) return Ty;
  return *const_cast<PATypeHolder*>(this) = NewTy;
}



//===----------------------------------------------------------------------===//
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
  return Ty.getTypeID() == Type::PointerTyID;
}

} // End llvm namespace

#endif
