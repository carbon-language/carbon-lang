//===-- llvm/DerivedTypes.h - Classes for handling data types ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of classes that represent "derived 
// types".  These are things like "arrays of x" or "structure of x, y, z" or
// "method returning x taking (y,z) as parameters", etc...
//
// The implementations of these classes live in the Type.cpp file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DERIVED_TYPES_H
#define LLVM_DERIVED_TYPES_H

#include "llvm/Type.h"

namespace llvm {

template<class ValType, class TypeClass> class TypeMap;
class FunctionValType;
class ArrayValType;
class StructValType;
class PointerValType;

class DerivedType : public Type, public AbstractTypeUser {
  /// RefCount - This counts the number of PATypeHolders that are pointing to
  /// this type.  When this number falls to zero, if the type is abstract and
  /// has no AbstractTypeUsers, the type is deleted.
  ///
  mutable unsigned RefCount;
  
  // AbstractTypeUsers - Implement a list of the users that need to be notified
  // if I am a type, and I get resolved into a more concrete type.
  //
  ///// FIXME: kill mutable nonsense when Types are not const
  mutable std::vector<AbstractTypeUser *> AbstractTypeUsers;

protected:
  DerivedType(PrimitiveID id) : Type("", id), RefCount(0) {}
  ~DerivedType() {
    assert(AbstractTypeUsers.empty());
  }

  /// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type
  /// that the current type has transitioned from being abstract to being
  /// concrete.
  ///
  void notifyUsesThatTypeBecameConcrete();

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.
  void dropAllTypeUses();
  
public:

  //===--------------------------------------------------------------------===//
  // Abstract Type handling methods - These types have special lifetimes, which
  // are managed by (add|remove)AbstractTypeUser. See comments in
  // AbstractTypeUser.h for more information.

  // addAbstractTypeUser - Notify an abstract type that there is a new user of
  // it.  This function is called primarily by the PATypeHandle class.
  //
  void addAbstractTypeUser(AbstractTypeUser *U) const {
    assert(isAbstract() && "addAbstractTypeUser: Current type not abstract!");
    AbstractTypeUsers.push_back(U);
  }

  // removeAbstractTypeUser - Notify an abstract type that a user of the class
  // no longer has a handle to the type.  This function is called primarily by
  // the PATypeHandle class.  When there are no users of the abstract type, it
  // is annihilated, because there is no way to get a reference to it ever
  // again.
  //
  void removeAbstractTypeUser(AbstractTypeUser *U) const;

  // refineAbstractTypeTo - This function is used to when it is discovered that
  // the 'this' abstract type is actually equivalent to the NewType specified.
  // This causes all users of 'this' to switch to reference the more concrete
  // type NewType and for 'this' to be deleted.
  //
  void refineAbstractTypeTo(const Type *NewType);

  void addRef() const {
    assert(isAbstract() && "Cannot add a reference to a non-abstract type!");
    ++RefCount;
  }

  void dropRef() const {
    assert(isAbstract() && "Cannot drop a refernce to a non-abstract type!");
    assert(RefCount && "No objects are currently referencing this object!");

    // If this is the last PATypeHolder using this object, and there are no
    // PATypeHandles using it, the type is dead, delete it now.
    if (--RefCount == 0 && AbstractTypeUsers.empty())
      delete this;
  }


  void dump() const { Value::dump(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const DerivedType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->isDerivedType();
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};




class FunctionType : public DerivedType {
  friend class TypeMap<FunctionValType, FunctionType>;
  bool isVarArgs;

  FunctionType(const FunctionType &);                   // Do not implement
  const FunctionType &operator=(const FunctionType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class FunctionType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  FunctionType(const Type *Result, const std::vector<const Type*> &Params, 
               bool IsVarArgs);

public:
  /// FunctionType::get - This static method is the primary way of constructing
  /// a FunctionType
  static FunctionType *get(const Type *Result,
                           const std::vector<const Type*> &Params,
                           bool isVarArg);

  inline bool isVarArg() const { return isVarArgs; }
  inline const Type *getReturnType() const { return ContainedTys[0]; }

  typedef std::vector<PATypeHandle>::const_iterator param_iterator;
  param_iterator param_begin() const { return ContainedTys.begin()+1; }
  param_iterator param_end() const { return ContainedTys.end(); }

  // Parameter type accessors...
  const Type *getParamType(unsigned i) const { return ContainedTys[i+1]; }

  // getNumParams - Return the number of fixed parameters this function type
  // requires.  This does not consider varargs.
  //
  unsigned getNumParams() const { return ContainedTys.size()-1; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FunctionType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == FunctionTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


// CompositeType - Common super class of ArrayType, StructType, and PointerType
//
class CompositeType : public DerivedType {
protected:
  inline CompositeType(PrimitiveID id) : DerivedType(id) { }
public:

  // getTypeAtIndex - Given an index value into the type, return the type of the
  // element.
  //
  virtual const Type *getTypeAtIndex(const Value *V) const = 0;
  virtual bool indexValid(const Value *V) const = 0;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CompositeType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == ArrayTyID || 
           T->getPrimitiveID() == StructTyID ||
           T->getPrimitiveID() == PointerTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


class StructType : public CompositeType {
  friend class TypeMap<StructValType, StructType>;
  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement

protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class StructType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  StructType(const std::vector<const Type*> &Types);

public:
  /// StructType::get - This static method is the primary way to create a
  /// StructType.
  static StructType *get(const std::vector<const Type*> &Params);

  // Iterator access to the elements
  typedef std::vector<PATypeHandle>::const_iterator element_iterator;
  element_iterator element_begin() const { return ContainedTys.begin(); }
  element_iterator element_end() const { return ContainedTys.end(); }

  // Random access to the elements
  unsigned getNumElements() const { return ContainedTys.size(); }
  const Type *getElementType(unsigned N) const {
    assert(N < ContainedTys.size() && "Element number out of range!");
    return ContainedTys[N];
  }

  // getTypeAtIndex - Given an index value into the type, return the type of the
  // element.  For a structure type, this must be a constant value...
  //
  virtual const Type *getTypeAtIndex(const Value *V) const ;
  virtual bool indexValid(const Value *V) const;

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StructType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == StructTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


// SequentialType - This is the superclass of the array and pointer type
// classes.  Both of these represent "arrays" in memory.  The array type
// represents a specifically sized array, pointer types are unsized/unknown size
// arrays.  SequentialType holds the common features of both, which stem from
// the fact that both lay their components out in memory identically.
//
class SequentialType : public CompositeType {
  SequentialType(const SequentialType &);                  // Do not implement!
  const SequentialType &operator=(const SequentialType &); // Do not implement!
protected:
  SequentialType(PrimitiveID TID, const Type *ElType) : CompositeType(TID) {
    ContainedTys.reserve(1);
    ContainedTys.push_back(PATypeHandle(ElType, this));
  }

public:
  inline const Type *getElementType() const { return ContainedTys[0]; }

  // getTypeAtIndex - Given an index value into the type, return the type of the
  // element.  For sequential types, there is only one subtype...
  //
  virtual const Type *getTypeAtIndex(const Value *V) const {
    return ContainedTys[0];
  }
  virtual bool indexValid(const Value *V) const {
    return V->getType()->isInteger();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SequentialType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == ArrayTyID ||
           T->getPrimitiveID() == PointerTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


class ArrayType : public SequentialType {
  friend class TypeMap<ArrayValType, ArrayType>;
  unsigned NumElements;

  ArrayType(const ArrayType &);                   // Do not implement
  const ArrayType &operator=(const ArrayType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class ArrayType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  ArrayType(const Type *ElType, unsigned NumEl);

public:
  /// ArrayType::get - This static method is the primary way to construct an
  /// ArrayType
  static ArrayType *get(const Type *ElementType, unsigned NumElements);

  inline unsigned    getNumElements() const { return NumElements; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ArrayType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == ArrayTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};



class PointerType : public SequentialType {
  friend class TypeMap<PointerValType, PointerType>;
  PointerType(const PointerType &);                   // Do not implement
  const PointerType &operator=(const PointerType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class PointerType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  PointerType(const Type *ElType);

public:
  /// PointerType::get - This is the only way to construct a new pointer type.
  static PointerType *get(const Type *ElementType);

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Implement support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PointerType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == PointerTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


class OpaqueType : public DerivedType {
  OpaqueType(const OpaqueType &);                   // DO NOT IMPLEMENT
  const OpaqueType &operator=(const OpaqueType &);  // DO NOT IMPLEMENT
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class OpaqueType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  OpaqueType();

public:
  // OpaqueType::get - Static factory method for the OpaqueType class...
  static OpaqueType *get() {
    return new OpaqueType();           // All opaque types are distinct
  }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    abort();   // FIXME: this is not really an AbstractTypeUser!
  }
  virtual void typeBecameConcrete(const DerivedType *AbsTy) {
    abort();   // FIXME: this is not really an AbstractTypeUser!
  }

  // Implement support for type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const OpaqueType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getPrimitiveID() == OpaqueTyID;
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};


// Define some inline methods for the AbstractTypeUser.h:PATypeHandle class.
// These are defined here because they MUST be inlined, yet are dependent on 
// the definition of the Type class.  Of course Type derives from Value, which
// contains an AbstractTypeUser instance, so there is no good way to factor out
// the code.  Hence this bit of uglyness.
//
inline void PATypeHandle::addUser() {
  assert(Ty && "Type Handle has a null type!");
  if (Ty->isAbstract())
    cast<DerivedType>(Ty)->addAbstractTypeUser(User);
}
inline void PATypeHandle::removeUser() {
  if (Ty->isAbstract())
    cast<DerivedType>(Ty)->removeAbstractTypeUser(User);
}

inline void PATypeHandle::removeUserFromConcrete() {
  if (!Ty->isAbstract())
    cast<DerivedType>(Ty)->removeAbstractTypeUser(User);
}

// Define inline methods for PATypeHolder...

inline void PATypeHolder::addRef() {
  if (Ty->isAbstract())
    cast<DerivedType>(Ty)->addRef();
}

inline void PATypeHolder::dropRef() {
  if (Ty->isAbstract())
    cast<DerivedType>(Ty)->dropRef();
}

/// get - This implements the forwarding part of the union-find algorithm for
/// abstract types.  Before every access to the Type*, we check to see if the
/// type we are pointing to is forwarding to a new type.  If so, we drop our
/// reference to the type.
inline const Type* PATypeHolder::get() const {
  const Type *NewTy = Ty->getForwardedType();
  if (!NewTy) return Ty;
  return *const_cast<PATypeHolder*>(this) = NewTy;
}

} // End llvm namespace

#endif
