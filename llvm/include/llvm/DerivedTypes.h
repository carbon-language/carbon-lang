//===-- llvm/DerivedTypes.h - Classes for handling data types ---*- C++ -*-===//
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

template<class ValType, class TypeClass> class TypeMap;
class FunctionValType;
class ArrayValType;
class StructValType;
class PointerValType;

class DerivedType : public Type {
  char isRefining;                                   // Used for recursive types

  // AbstractTypeUsers - Implement a list of the users that need to be notified
  // if I am a type, and I get resolved into a more concrete type.
  //
  ///// FIXME: kill mutable nonsense when Type's are not const
  mutable std::vector<AbstractTypeUser *> AbstractTypeUsers;

protected:
  inline DerivedType(PrimitiveID id) : Type("", id) {
    isRefining = 0;
  }
  ~DerivedType() {
    assert(AbstractTypeUsers.empty());
  }

  // typeIsRefined - Notify AbstractTypeUsers of this type that the current type
  // has been refined a bit.  The pointer is still valid and still should be
  // used, but the subtypes have changed.
  //
  void typeIsRefined();

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.  This also removes the
  // type from the internal tables of available types.
  virtual void dropAllTypeUses(bool inMap) = 0;
  

  void refineAbstractTypeToInternal(const Type *NewType, bool inMap);

public:

  //===--------------------------------------------------------------------===//
  // Abstract Type handling methods - These types have special lifetimes, which
  // are managed by (add|remove)AbstractTypeUser. See comments in
  // AbstractTypeUser.h for more information.

  // addAbstractTypeUser - Notify an abstract type that there is a new user of
  // it.  This function is called primarily by the PATypeHandle class.
  //
  void addAbstractTypeUser(AbstractTypeUser *U) const;

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
  void refineAbstractTypeTo(const Type *NewType) {
    refineAbstractTypeToInternal(NewType, true);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const DerivedType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->isDerivedType();
  }
  static inline bool classof(const Value *V) {
    return isa<Type>(V) && classof(cast<Type>(V));
  }
};




struct FunctionType : public DerivedType {
  typedef std::vector<PATypeHandle> ParamTypes;
  friend class TypeMap<FunctionValType, FunctionType>;
private:
  PATypeHandle ResultType;
  ParamTypes ParamTys;
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

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.  This also removes the
  // type from the internal tables of available types.
  virtual void dropAllTypeUses(bool inMap);

public:

  inline bool isVarArg() const { return isVarArgs; }
  inline const Type *getReturnType() const { return ResultType; }
  inline const ParamTypes &getParamTypes() const { return ParamTys; }

  // Parameter type accessors...
  const Type *getParamType(unsigned i) const { return ParamTys[i]; }

  // getNumParams - Return the number of fixed parameters this function type
  // requires.  This does not consider varargs.
  //
  unsigned getNumParams() const { return ParamTys.size(); }


  virtual const Type *getContainedType(unsigned i) const {
    return i == 0 ? ResultType : 
                    (i <= ParamTys.size() ? ParamTys[i-1].get() : 0);
  }
  virtual unsigned getNumContainedTypes() const { return ParamTys.size()+1; }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static FunctionType *get(const Type *Result,
                           const std::vector<const Type*> &Params,
                           bool isVarArg);


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

  // getIndexType - Return the type required of indices for this composite.
  // For structures, this is ubyte, for arrays, this is uint
  //
  virtual const Type *getIndexType() const = 0;


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


struct StructType : public CompositeType {
  friend class TypeMap<StructValType, StructType>;
  typedef std::vector<PATypeHandle> ElementTypes;

private:
  ElementTypes ETypes;                              // Element types of struct

  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement

protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class StructType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  StructType(const std::vector<const Type*> &Types);

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.  This also removes the
  // type from the internal tables of available types.
  virtual void dropAllTypeUses(bool inMap);
  
public:
  inline const ElementTypes &getElementTypes() const { return ETypes; }

  virtual const Type *getContainedType(unsigned i) const { 
    return i < ETypes.size() ? ETypes[i].get() : 0;
  }
  virtual unsigned getNumContainedTypes() const { return ETypes.size(); }

  // getTypeAtIndex - Given an index value into the type, return the type of the
  // element.  For a structure type, this must be a constant value...
  //
  virtual const Type *getTypeAtIndex(const Value *V) const ;
  virtual bool indexValid(const Value *V) const;

  // getIndexType - Return the type required of indices for this composite.
  // For structures, this is ubyte, for arrays, this is uint
  //
  virtual const Type *getIndexType() const { return Type::UByteTy; }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static StructType *get(const std::vector<const Type*> &Params);

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
  PATypeHandle ElementType;

  SequentialType(PrimitiveID TID, const Type *ElType)
    : CompositeType(TID), ElementType(PATypeHandle(ElType, this)) {
  }

public:
  inline const Type *getElementType() const { return ElementType; }

  virtual const Type *getContainedType(unsigned i) const { 
    return i == 0 ? ElementType.get() : 0;
  }
  virtual unsigned getNumContainedTypes() const { return 1; }

  // getTypeAtIndex - Given an index value into the type, return the type of the
  // element.  For sequential types, there is only one subtype...
  //
  virtual const Type *getTypeAtIndex(const Value *V) const {
    return ElementType.get();
  }
  virtual bool indexValid(const Value *V) const {
    return V->getType() == Type::LongTy;   // Must be a 'long' index
  }

  // getIndexType() - Return the type required of indices for this composite.
  // For structures, this is ubyte, for arrays, this is uint
  //
  virtual const Type *getIndexType() const { return Type::LongTy; }

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

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.  This also removes the
  // type from the internal tables of available types.
  virtual void dropAllTypeUses(bool inMap);

public:
  inline unsigned    getNumElements() const { return NumElements; }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static ArrayType *get(const Type *ElementType, unsigned NumElements);

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

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.  This also removes the
  // type from the internal tables of available types.
  virtual void dropAllTypeUses(bool inMap);
public:
  // PointerType::get - Named constructor for pointer types...
  static PointerType *get(const Type *ElementType);

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
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

  // dropAllTypeUses - When this (abstract) type is resolved to be equal to
  // another (more concrete) type, we must eliminate all references to other
  // types, to avoid some circular reference problems.
  virtual void dropAllTypeUses(bool inMap) {}  // No type uses

public:

  // get - Static factory method for the OpaqueType class...
  static OpaqueType *get() {
    return new OpaqueType();           // All opaque types are distinct
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
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

#endif
